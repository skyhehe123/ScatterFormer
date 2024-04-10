import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from math import ceil, floor

from pcdet.models.model_utils.scatterformer_utils import ScatterAttention

from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned

import spconv.pytorch as spconv
import torch.nn.functional as F

from mmcv.ops.masked_conv import ext_module

class ScatterFormer(nn.Module):
    '''Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in set attention for each stage. 
            Length: stage num.
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages. One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.

    '''
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        win_size = self.model_cfg.win_size
        block_name = self.model_cfg.block_name
        attn_type = self.model_cfg.attn_type
        depth = self.model_cfg.depth
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
   
        # save GPU memory
        self.use_torch_ckpt = self.model_cfg.get('USE_CHECKPOINT', False)
 
        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            ws_this_stage = win_size[stage_id]
            dmodel_this_stage = d_model[stage_id]
            num_blocks_this_stage = depth[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_name_this_stage = block_name[stage_id]
            block_module = _get_block_module(block_name_this_stage)
            block_list=[]
            norm_list=[]
            for i in range(num_blocks_this_stage):
                block_list.append( 
                    block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage, dropout, activation, attn_type, ws_this_stage) 
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

        self.win_size = self.model_cfg.win_size 
        self.output_shape = self.model_cfg.output_shape
        self.stage_num = stage_num
        self.num_point_features = self.model_cfg.conv_out_channel

        self._reset_parameters()

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        '''      
        block_id = 0
        for stage_id in range(self.stage_num):
            ws = self.win_size[stage_id]

            output = batch_dict['voxel_features']
            coors = batch_dict['voxel_coords'][:, [0, 3, 2]].int()
            batch_win_coords = torch.cat([ batch_dict['voxel_coords'][:, :1], coors[:, 1:] // ws ], dim=1) 
            win_pos = ( coors[:, 1:] % ws ).long()
            _, batch_win_inds = torch.unique(batch_win_coords, return_inverse=True, dim=0)

            batch_win_inds, perm = torch.sort(batch_win_inds)
            counts = torch.bincount(batch_win_inds)
            offsets = F.pad( torch.cumsum(counts, dim=0), (1,0), mode='constant', value=0)[:-1]

            output = output[perm]
            coors = coors[perm]
            win_pos = win_pos[perm]
            batch_dict['voxel_coords'] = batch_dict['voxel_coords'][perm]


            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                if self.use_torch_ckpt == False or not self.training:
                    output = block(
                        output, offsets, counts, batch_win_inds, win_pos,
                        coors, self.output_shape, batch_dict['batch_size'], block_id)
                else:
                    output = checkpoint(block, 
                        output, offsets, counts, batch_win_inds, win_pos,
                        coors, self.output_shape, batch_dict['batch_size'], block_id
                    )
                output = residual_norm_layers[i](output + residual)
                block_id += 1
            

            batch_dict['pillar_features'] = batch_dict['voxel_features'] = output
        
        return batch_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

class CWILayer(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu", kernel_size=13):
        super().__init__()
       
        self.d_model = d_model
        self.padding = kernel_size // 2
        self.gc = int(d_model * .25)
        self.dwconv_hw =nn.Conv2d(self.gc, self.gc, 3, padding=3//2, groups=self.gc) 
        self.dwconv_w = nn.Conv2d(self.gc, self.gc, kernel_size=(1, kernel_size), padding=(0, self.padding), groups=self.gc) 
        self.dwconv_h = nn.Conv2d(self.gc, self.gc, kernel_size=(kernel_size, 1), padding=(self.padding, 0), groups=self.gc) 
        self.split_indexes = (d_model - 3 * self.gc, self.gc, self.gc, self.gc)

        # MLP
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = _get_activation_fn(activation)

    def forward( self,src, coords, spatial_shape, batch_size):
        
        x_id, x_hw, x_w, x_h = torch.split(src, self.split_indexes, dim=1)

        x_hw = spconv.SparseConvTensor(x_hw, coords, spatial_shape, batch_size).dense()
        x_w = spconv.SparseConvTensor(x_w, coords, spatial_shape, batch_size).dense()
        x_h = spconv.SparseConvTensor(x_h, coords, spatial_shape, batch_size).dense()
        
        if self.training or batch_size != 1:
            flatten_indices = (coords[:, 0] * spatial_shape[0] * spatial_shape[1] \
                + coords[:, 1] * spatial_shape[1] + coords[:, 2]).long()
            x_hw = self.dwconv_hw(x_hw).permute(0,2,3,1).contiguous().view(-1, self.gc)
            x_w = self.dwconv_w(x_w).permute(0,2,3,1).contiguous().view(-1, self.gc)
            x_h = self.dwconv_h(x_h).permute(0,2,3,1).contiguous().view(-1, self.gc)
            src = torch.cat([x_id, x_hw[flatten_indices, :], x_w[flatten_indices, :], x_h[flatten_indices, :]], dim=1)
        else:
            assert batch_size == 1
            coords_xy = coords[:, 1:].long()
            x_hw = sparse_group_conv2d(x_hw, coords_xy, self.dwconv_hw.weight, self.dwconv_hw.bias, (1, 1))
            x_w = sparse_group_conv2d(x_w, coords_xy, self.dwconv_w.weight, self.dwconv_w.bias, (0, self.padding))
            x_h = sparse_group_conv2d(x_h, coords_xy, self.dwconv_h.weight, self.dwconv_h.bias, (self.padding, 0))
            src = torch.cat([x_id, x_hw, x_w, x_h], dim=1)

       
        return  self.linear2(self.dropout(self.activation(self.linear1(self.norm(src)))))
      
    
class ScatterFormerBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", attn_type='xcit', win_size=12):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.win_size = win_size
        self.pe = nn.Embedding(win_size**2, d_model)
        self.attn = ScatterAttention(d_model, nhead, attn_type)
        self.norm0 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
       
        self.conv = CWILayer(d_model, dim_feedforward, dropout, activation, kernel_size=win_size + 1)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def forward(
            self,
            src,
            offsets, 
            counts, 
            batch_win_inds, 
            win_pos,
            coords, 
            spatial_shape, 
            batch_size, 
            block_id,
    ):
        
        pe = self.pe(win_pos[:, 0] * self.win_size + win_pos[:, 1])
        src2 = self.attn(src, offsets, counts, batch_win_inds, pe=pe)
        src = src + src2
        src = self.norm0(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm1(src)

        # CWI
        src2 = self.conv(src, coords, spatial_shape, batch_size)
        src = src + src2
        src = self.norm2(src)

        return src
    

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_block_module(name):
    """Return an block module given a string"""
    if name == "ScatterFormerBlock":
        return ScatterFormerBlock
    raise RuntimeError(F"This Block not exist.")





    
def sparse_group_conv2d(
            features: torch.Tensor,
            coors: torch.Tensor,
            weight: torch.nn.Parameter,
            bias: torch.nn.Parameter,
            pad_size: tuple) -> torch.Tensor:
 
    out_channel, _, kernel_h, kernel_w = weight.size()
    pad_h, pad_w = pad_size   

    data_col = features.new_zeros(out_channel, kernel_h * kernel_w, coors.size(0))
    ext_module.masked_im2col_forward(
        features,
        coors[:, 0].contiguous(),
        coors[:, 1].contiguous(),
        data_col,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        pad_h=pad_h,
        pad_w=pad_w)
    w = weight.view(out_channel, kernel_h * kernel_w)
    output = torch.einsum('ckn,ck->nc', data_col, w) + bias[None, :]
      
    return output

