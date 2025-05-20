#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template <typename T>
__global__ void forward_sparse_dwconv_kernel(
    T* out_features,
    const T* in_features,
    const T* weights,
    const int* indices,
    int num_features,
    int RS,
    int num_indices) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= num_indices || feature_idx >= num_features) return;

    // Compute weighted average for this feature
    T sum = 0;
    for (int k = 0; k < RS; ++k) {
        int in_idx = indices[k * num_indices + idx];
        if (in_idx != -1) {
            sum += in_features[in_idx * num_features + feature_idx] * weights[k * num_features + feature_idx];
        }
    }
    out_features[idx * num_features + feature_idx] = sum;
}


// Backward kernel with warp-level reduction
template <typename T>
__global__ void backward_sparse_dwconv_kernel(
    const T* __restrict__ dout_features,
    const T* __restrict__ in_features,
    const T* __restrict__ weights,
    T* __restrict__ din_features,
    T* __restrict__ dw_features,
    const int* __restrict__ indices,
    int num_features,
    int RS,
    int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= num_indices || feature_idx >= num_features) return;

    // Compute grad_input
    T grad_input = 0;
    for (int k = 0; k < RS; ++k) {
        int in_idx = indices[k * num_indices + idx];
        if (in_idx >= 0) {
            T go = dout_features[in_idx * num_features + feature_idx];
            T w  = weights[k * num_features + feature_idx];
            grad_input += go * w;
        }
    }
    din_features[idx * num_features + feature_idx] = grad_input;

    // Warp-level reduction for weight gradients
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned mask = 0xffffffff;
    int lane = linear_tid & 31;

    for (int k = 0; k < RS; ++k) {
        int in_idx = indices[k * num_indices + idx];
        T partial = T(0);
        if (in_idx >= 0) {
            T go = dout_features[in_idx * num_features + feature_idx];
            T iv = in_features[idx * num_features + feature_idx];
            partial = go * iv;
        }
        // Shuffle-based warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }
        if (lane == 0) {
            atomicAdd(&dw_features[k * num_features + feature_idx], partial);
        }
    }
}

// Forward wrapper
torch::Tensor indice_sparse_dwconv(
    torch::Tensor features,
    torch::Tensor weights,
    torch::Tensor indices) {
    int num_indices = indices.size(1);
    int num_features = features.size(1);
    int RS = indices.size(0);

    auto out_features = torch::zeros({num_indices, num_features}, features.options());

    // Adjust block size for better memory access pattern
    dim3 threads(16, 16);  // Changed to better match warp size
    dim3 blocks((num_indices + threads.x - 1) / threads.x,
                (num_features + threads.y - 1) / threads.y);
  
    AT_DISPATCH_FLOATING_TYPES(features.scalar_type(), "forward_sparse_dwconv", [&] {
        forward_sparse_dwconv_kernel<scalar_t><<<blocks, threads>>>(
            out_features.data_ptr<scalar_t>(),
            features.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            indices.data_ptr<int>(),
            num_features,
            RS,
            num_indices);
    });

    return out_features;
}

// Backward wrapper
std::vector<torch::Tensor> indice_sparse_dwconv_backward(
    torch::Tensor dout_features,
    torch::Tensor in_features,
    torch::Tensor weights,
    torch::Tensor indices) {
    int num_indices = indices.size(1);
    int num_features = in_features.size(1);
    int RS = indices.size(0);

    auto din_features = torch::zeros_like(in_features);
    auto dw_features = torch::zeros_like(weights);

    dim3 threads(32, 8);  // 32 threads in x-dimension for warp reduction
    dim3 blocks((num_indices + threads.x - 1) / threads.x,
                (num_features + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(dout_features.scalar_type(), "backward_sparse_dwconv", [&] {
        backward_sparse_dwconv_kernel<scalar_t><<<blocks, threads>>>(
            dout_features.data_ptr<scalar_t>(),
            in_features.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            din_features.data_ptr<scalar_t>(),
            dw_features.data_ptr<scalar_t>(),
            indices.data_ptr<int>(),
            num_features,
            RS,
            num_indices);
    });

    return {din_features, dw_features};
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("indice_sparse_dwconv", &indice_sparse_dwconv,
          "Sparse DWConv Forward");
    m.def("indice_sparse_dwconv_backward", &indice_sparse_dwconv_backward,
          "Sparse DWConv Backward");
}