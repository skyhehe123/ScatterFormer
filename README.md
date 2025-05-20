# ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention
	

This repo is the official implementation of paper: [ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention](https://arxiv.org/pdf/2401.00912). It achieves 73.8 mAP L2 on Waymo Open Dataset val and 72.4 NDS on NuScenes val. The scatterformer achieve real-time speed of 23 FPS.

<!-- > ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention
>
> [Chenhang He*](https://skyhehe123.github.io/), Ruihuang Li, Guowen Zhang, Lei Zhang -->


<div align="center">
  <img src="assets/performance.png" width="400"/>
</div>

## News
- [24-06-21] Scatterformer is accepted by ECCV 2024!  
- [24-07-18] Training code released 
- [25-05-20] Added CUDA implementation for group-wise sparse convolution


## Main results

### Waymo Open Dataset validation
|  Model  |  #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1 | Veh_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Log |
|---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
|  [ScatterFormer (100%)](tools/cfgs/waymo_models/scatterformer.yaml) | 1       |  81.8/79.7  | 75.7/73.8  | 81.0/80.5 | 73.1/72.7 | 84.5/79.9 | 77.0/72.6 | 79.9/78.9 | 77.1/76.1 | [Log](https://drive.google.com/file/d/1WamAN8tBRg8aq35Ia6PsEdkYo-jxKNS1/view?usp=sharing) |
|  [ScatterFormer (20%)](tools/cfgs/waymo_models/scatterformer.yaml) | 1       |  80.3/78.0  | 74.1/72.0  | 79.6/79.1 | 71.6/71.2 | 83.5/78.3 | 75.9/71.0 | 77.7/76.6 | 74.8/73.7 | [Log](https://drive.google.com/file/d/1_nGyTDCGy8oo-tkw7BHKoO_w8eMzPbm8/view?usp=sharing) |
### NuScenes validation
|  Model  | mAP | NDS | mATE | mASE | mAOE | mAVE| mAAE | ckpt | Log |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|--------|
|  [ScatterFormer](tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences.yaml) | 68.3 | 72.4 | 26.5 | 24.5 | 24.7 | 23.3 | 18.8| [ckpt](https://drive.google.com/file/d/1AJp0EQoXw-8JNI98SkD-k1DqLzKHYNJK/view?usp=sharing)| [Log](https://drive.google.com/file/d/1kiDoCiu8YzIyy5t_DM5XCAMKa3OR-2wh/view?usp=sharing)| 



## Usage
### Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation.

### Dataset Preparation
Please follow the instructions from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). We adopt the same data generation process.

### Sparse Group-wise Convolution
~~ScatterFormer relies on a group-wise sparse convolution, please find 
this hacked version of [spconv]
(https://github.com/skyhehe123/spconv)~~

We update the cuda implementation of group-wise sparse convolution in `pcdet/ops/dw_spconv/`. You need to compile the CUDA code:

```bash
cd pcdet/ops/dw_spconv
python setup.py build_ext --inplace
```

This will compile the CUDA implementation and you won't need to recompile the hacked version of spconv anymore.

### Training
```
# multi-gpu training
cd tools
bash scripts/dist_train.sh 8 --cfg_file <CONFIG_FILE> [other optional arguments]
```

### Testing
```
# multi-gpu testing
cd tools
bash scripts/dist_test.sh 8 --cfg_file <CONFIG_FILE> --ckpt <CHECKPOINT_FILE>
```

