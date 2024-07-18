import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from torch import Tensor

import triton
import triton.language as tl
import torch.nn as nn
from einops import rearrange
import torch_scatter
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based import surrogate, neuron
class LIF(neuron.LIFNode):
    def __init__(self):
        super().__init__(tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)
        
GROUP_SIZE = 32

@triton.jit
def scatter_relu_attention_fwd_kernel(
    Q, K, V, 
    offsets, counts, O, 
    h:tl.constexpr, 
    d:tl.constexpr, 
    g:tl.constexpr
):
  
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count_n = tl.load( counts + pid )

    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    

    for delta_v in range(0, count_n, g):
       
        offs_v =(start + delta_v) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask_v = (delta_v + idx_x)[:, None] < count_n
        k = tl.load(K+offs_v, mask=mask_v, other=0.0)
        v = tl.load(V+offs_v, mask=mask_v, other=0.0)
        
        for delta_q in range(0, count_n, g):
            offs =(start + delta_q) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
            mask = (delta_q + idx_x)[:, None] < count_n
            q = tl.load(Q+offs, mask=mask, other=0.0)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False) 
          
            relu_qk = tl.where(qk >= 0, qk, 0.) 
            qk = relu_qk / count_n
            
            qkv = tl.dot(qk, v, allow_tf32=False)
            
            tl.atomic_add(O + offs, qkv, mask=mask)
        
        
@triton.jit      
def scatter_relu_attention_bwd_kernel(mat_q, mat_k, mat_v, offsets, counts, dout, dmat_q, dmat_k, dmat_v, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count = tl.load( counts + pid )
    
    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)

    for delta_v in range(0, count, g):
        offs_v =(start + delta_v) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask_v = (delta_v + idx_x)[:, None] < count
        k = tl.load(mat_k+offs_v, mask=mask_v, other=0.0)
        v = tl.load(mat_v+offs_v, mask=mask_v, other=0.0)
        
        for delta_q in range(0, count, g):
            offs =(start + delta_q) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
            mask = (delta_q + idx_x)[:, None] < count
            q = tl.load(mat_q + offs, mask=mask, other=0.0)
            do = tl.load(dout + offs, mask=mask, other=0.0)

            qk = tl.dot(q, tl.trans(k), allow_tf32=False) 
            relu_qk = tl.where(qk >= 0, qk, 0.) 
            qk = relu_qk / count
            
            # Backward pass
            dqk = tl.dot(do, tl.trans(v), allow_tf32=False) 
            dqk = tl.where(relu_qk > 0, dqk, 0.) 
            dqk = dqk / count
         
            dq = tl.dot(dqk, k, allow_tf32=False)
            dk = tl.dot(tl.trans(dqk), q, allow_tf32=False)
            dv = tl.dot(tl.trans(qk), do, allow_tf32=False)   
            
            # Gradient accumulation
            tl.atomic_add(dmat_q + offs, dq, mask=mask)
            tl.atomic_add(dmat_k + offs_v, dk, mask=mask_v)
            tl.atomic_add(dmat_v + offs_v, dv, mask=mask_v)


class ScatterReLUAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, offsets, counts):
        
        n, h, d = q.shape

        if not q.is_contiguous():
            q = q.contiguous()
      
        if not k.is_contiguous():
            k = k.contiguous()

        if not v.is_contiguous():
            v = v.contiguous()

        m = len(offsets)
        
        o = torch.zeros([n, h, d], dtype=k.dtype, device=v.device)
       
        scatter_relu_attention_fwd_kernel[(m, h)](
            q, k, v, offsets, counts, o, 
            h=h, d=d, g=GROUP_SIZE
        )
        
        ctx.save_for_backward(q, k, v, offsets, counts)

        ctx.h = h 
        ctx.d = d
        ctx.m = m
       

        return o
    
    @staticmethod
    def backward(ctx, dout):
        mat_q, mat_k, mat_v, offsets, counts = ctx.saved_tensors
        dq = torch.zeros_like(mat_q, dtype=torch.float32)
        dk = torch.zeros_like(mat_k, dtype=torch.float32)
        dv = torch.zeros_like(mat_v, dtype=torch.float32)
      
        d = ctx.d
        m = ctx.m
        h = ctx.h
      
        scatter_relu_attention_bwd_kernel[(m, h)](
             mat_q, mat_k, mat_v, offsets, counts, dout.contiguous(), dq, dk, dv, h=h, d=d, g=GROUP_SIZE
        )
    
        return dq, dk, dv, None, None

scatter_relu_attention = ScatterReLUAttention.apply



class EfficientMultiheadAttention(nn.MultiheadAttention):

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            seq_len: int = 1,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
      
        is_batched = query.dim() == 3
    
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        
        attn_output, attn_output_weights = self.efficient_multi_head_attention_forward(
                query, key, value, seq_len, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def efficient_multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_len: int,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
    
        is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # key_padding_mask = F._canonical_mask(
        #     mask=key_padding_mask,
        #     mask_name="key_padding_mask",
        #     other_type= F._none_or_dtype(attn_mask),
        #     other_name="attn_mask",
        #     target_type=query.dtype
        # )
        key_padding_mask = torch.arange(src_len).to(seq_len.device). \
            expand(len(seq_len), src_len) < seq_len.unsqueeze(1)
        
        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
       
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
       
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
      
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
       
        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            attn_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        
        if seq_len is not None:
            scale_map = seq_len.view(bsz, 1).expand(-1, num_heads).reshape(bsz * num_heads, 1, 1)

        q_scaled = q / math.sqrt(q.shape[-1])

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1)) 
       
        attn_output_weights = F.relu(attn_output_weights) / scale_map
        

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
       
    def flash_forward(self, q, k, v, counts):
      
        offsets = F.pad( torch.cumsum(counts, dim=0), (1,0), mode='constant', value=0)[:-1]
        
        q, k, v = F._in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)

        q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])
       
        q_scaled = q / math.sqrt(self.head_dim)

        attn_output = scatter_relu_attention(q_scaled, k, v, offsets, counts)
      
        attn_output = rearrange(attn_output, "n h c -> n (h c)", h=self.num_heads)
   
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        
        return attn_output
    

      
if __name__ == "__main__":
    batch_win_inds = torch.load('batch_win_inds')

    counts = torch.bincount(batch_win_inds)
    offsets = F.pad( torch.cumsum(counts, dim=0), (1,0), mode='constant', value=0)[:-1]

    N = len(batch_win_inds)

    q = torch.randn(N, 6, 32).cuda().requires_grad_(True)
    k = torch.randn(N, 6, 32).cuda().requires_grad_(True)
    v = torch.randn(N, 6, 32).cuda().requires_grad_(True)
    


    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    for _ in range(10000):

        # start.record()
        
        output = list()
        for i in range(int(batch_win_inds.max())+1):
            q_ = q[batch_win_inds==i, ...]
            k_ = k[batch_win_inds==i, ...]
            v_ = v[batch_win_inds==i, ...]
            attn = torch.einsum('nhd, mhd->hnm',q_, k_)
            attn = torch.relu(attn) / len(q_)
            output.append( torch.einsum('hnm, mhd->nhd', attn, v_))
        output = torch.cat(output, dim=0)
        
        gq = torch.autograd.grad(output.sum(), q, retain_graph=True)[0]
        gk = torch.autograd.grad(output.sum(), k, retain_graph=True)[0]
        gv = torch.autograd.grad(output.sum(), v, retain_graph=True)[0]

        output_triton = scatter_relu_attention(q, k, v, offsets, counts)

        gq_triton = torch.autograd.grad(output_triton.sum(), q, retain_graph=True)[0]
        gk_triton = torch.autograd.grad(output_triton.sum(), k, retain_graph=True)[0]
        gv_triton = torch.autograd.grad(output_triton.sum(), v, retain_graph=True)[0]

        # end.record()

        # # Waits for everything to finish running
        # torch.cuda.synchronize()

        # print(start.elapsed_time(end))
        print((output - output_triton).max())
        print((gq - gq_triton).max())
        print((gk - gk_triton).max())
        print((gv - gv_triton).max())