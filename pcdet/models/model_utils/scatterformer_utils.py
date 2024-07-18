import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch.nn as nn
from einops import rearrange
import torch_scatter
import spconv.pytorch as spconv

BLOCK_SIZE=32

@triton.jit
def _scatter_kv_fused_fwd_kernel(mat_k, mat_v, offsets, counts, kv, s, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):
    
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count = tl.load( counts + pid )
    
    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    idx_y = pid * d * d * h + hid * d * d + idx_d[:, None] * d + idx_d[None, :]
    idx_s = pid * d * h + hid * d + idx_d
    cum_kv = tl.zeros([d, d], dtype=tl.float32)
    cum_k = tl.zeros([d], dtype=tl.float32)
   
    for delta in range(0, count, g):
        offs =(start + delta) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask = (delta + idx_x)[:, None] < count
        k = tl.load(mat_k+offs, mask=mask, other=0.0)
        v = tl.load(mat_v+offs, mask=mask, other=0.0)
        cum_kv += tl.dot(tl.trans(k), v, allow_tf32=False)
        cum_k += tl.sum(k, 0)
       
    tl.store(kv + idx_y, cum_kv)
    tl.store(s + idx_s, cum_k)

@triton.jit
def _scatter_qc_fused_fwd_kernel(mat_q, mat_c, mat_s, offsets, counts, out, z, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):
    
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count = tl.load( counts + pid )
    
    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    idx_y = pid * d * d * h + hid * d * d + idx_d[:, None] * d + idx_d[None, :]
    idx_s = pid * d * h + hid * d + idx_d
    
    c = tl.load(mat_c + idx_y)
    s = tl.load(mat_s + idx_s)
    
    for delta in range(0, count, g):
        offs =(start + delta) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask = (delta + idx_x)[:, None] < count
        q = tl.load(mat_q + offs, mask=mask, other=0.0)
        y = tl.dot(q, c, allow_tf32=False)
        tl.store(out + offs, y, mask=mask)
        qs = tl.sum( q * s[None, :], 1)
        tl.store(z + (start + delta) * h + idx_x[:, None] * h + hid, qs[:, None], mask=mask)

@triton.jit
def _scatter_kv_fwd_kernel(mat_k, mat_v, offsets, counts, out, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):
    
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count = tl.load( counts + pid )
    
    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    idx_y = pid * d * d * h + hid * d * d + idx_d[:, None] * d + idx_d[None, :]
    
    cum_kv = tl.zeros([d, d], dtype=tl.float32)
    for delta in range(0, count, g):
        offs =(start + delta) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask = (delta + idx_x)[:, None] < count
        k = tl.load(mat_k+offs, mask=mask, other=0.0)
        v = tl.load(mat_v+offs, mask=mask, other=0.0)
        cum_kv += tl.dot(tl.trans(k), v, allow_tf32=False)
        
    tl.store(out + idx_y, cum_kv)

@triton.jit    
def _scatter_kv_bwd_kernel(dout, mat_k, mat_v, offsets, counts, dk, dv, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):

    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load(offsets + pid)
    count = tl.load(counts + pid)


    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    idx_y = pid * d * d * h + hid * d * d + idx_d[:, None] * d + idx_d[None, :]

    dy = tl.load(dout + idx_y)
    for delta in range(0, count, g):
        offs =(start + delta) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask = (delta + idx_x)[:, None] < count
        k = tl.load(mat_k+offs, mask=mask, other=0.0)
        v = tl.load(mat_v+offs, mask=mask, other=0.0)
        gv = tl.dot(k, dy, allow_tf32=False) 
        gk = tl.dot(v, tl.trans(dy), allow_tf32=False)
        tl.store(dv + offs, gv, mask=mask)
        tl.store(dk + offs, gk, mask=mask)




@triton.jit
def _scatter_qc_fwd_kernel(mat_q, mat_c, offsets, counts, out, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):
    
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count = tl.load( counts + pid )
    
    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    idx_y = pid * d * d * h + hid * d * d + idx_d[:, None] * d + idx_d[None, :]
    
    c = tl.load(mat_c + idx_y)
    
    for delta in range(0, count, g):
        offs =(start + delta) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask = (delta + idx_x)[:, None] < count
        q = tl.load(mat_q + offs, mask=mask, other=0.0)
        y = tl.dot(q, c, allow_tf32=False)
        tl.store(out + offs, y, mask=mask)

@triton.jit
def _scatter_qc_bwd_kernel(dout, mat_q, mat_c, offsets, counts, dq, dc, h:tl.constexpr, d:tl.constexpr, g:tl.constexpr):
    
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    start = tl.load( offsets + pid )
    count = tl.load( counts + pid )  
  
    idx_d = tl.arange(0, d)
    idx_x = tl.arange(0, g)
    idx_y = pid * d * d * h + hid * d * d + idx_d[:, None] * d + idx_d[None, :]

    c = tl.load(mat_c + idx_y)

    for delta in range(0, count, g):
        offs =(start + delta) * d * h + idx_x[:, None] * d * h + hid * d + idx_d[None, :]
        mask = (delta + idx_x)[:, None] < count
        dy = tl.load(dout + offs , mask=mask, other=0.0)
        q = tl.load(mat_q + offs, mask=mask, other=0.0)
       
        gq = tl.dot(dy, tl.trans(c), allow_tf32=False) 
        gc = tl.dot(tl.trans(q), dy, allow_tf32=False)

        tl.store(dq + offs, gq, mask=mask)
        tl.atomic_add(dc + idx_y, gc)


def scatter_kv_fused(mat_k, mat_v, offsets, counts):
    
    n, h, d = mat_k.shape
  
    if not mat_k.is_contiguous():
        mat_k = mat_k.contiguous()
    if not mat_v.is_contiguous():
        mat_v = mat_v.contiguous()
    m = len(offsets)
    
    kv = torch.zeros([m, h, d, d], dtype=mat_k.dtype, device=mat_k.device)
    s = torch.zeros([m, h, d], dtype=mat_k.dtype, device=mat_k.device)
    _scatter_kv_fused_fwd_kernel[(m, h)](
        mat_k, mat_v, offsets, counts, kv, s, h=h, d=d, g=BLOCK_SIZE
    )
    return kv, s


def scatter_qc_fused(mat_q, mat_c, mat_s, offsets, counts):
    
    n, h, d = mat_q.shape
  
    if not mat_q.is_contiguous():
        mat_q = mat_q.contiguous()
    if not mat_c.is_contiguous():
        mat_c = mat_c.contiguous()
    
    if not mat_s.is_contiguous():
        mat_s = mat_s.contiguous()

    m = len(offsets)
    
    out = torch.zeros([n, h, d], dtype=mat_q.dtype, device=mat_q.device)
    z = torch.zeros([n, h, 1], dtype=mat_q.dtype, device=mat_q.device)
    _scatter_qc_fused_fwd_kernel[(m, h)](
        mat_q, mat_c, mat_s, offsets, counts, out, z, h=h, d=d, g=BLOCK_SIZE
    )
    return out, z
    
class ScatterMatmulKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat_k, mat_v, offsets, counts):
        
        n, h, d = mat_k.shape
      
        if not mat_k.is_contiguous():
            mat_k = mat_k.contiguous()

        if not mat_v.is_contiguous():
            mat_v = mat_v.contiguous()

        m = len(offsets)
        
        kv = torch.zeros([m, h, d, d], dtype=mat_k.dtype, device=mat_k.device)
  
        
        _scatter_kv_fwd_kernel[(m, h)](
            mat_k, mat_v, offsets, counts, kv, h=h, d=d, g=BLOCK_SIZE
        )
        
        ctx.save_for_backward(mat_k, mat_v, offsets, counts)
        
        ctx.d = d
        ctx.m = m
        ctx.h = h

        return kv
    
    @staticmethod
    def backward(ctx, dout):
         
        mat_k, mat_v, offsets, counts = ctx.saved_tensors
        dk = torch.zeros_like(mat_k, dtype=torch.float32)
        dv = torch.zeros_like(mat_v, dtype=torch.float32)
        h = ctx.h
        d = ctx.d
        m = ctx.m
        
        _scatter_kv_bwd_kernel[(m, h)](
            dout.contiguous(), mat_k, mat_v, offsets, counts, dk, dv, h=h, d=d, g=BLOCK_SIZE
        )
       
        return dk, dv, None, None

scatter_matmul_kv = ScatterMatmulKV.apply


class ScatterMatmulQC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mat_q, mat_c, offsets, counts):
        
        n, h, d = mat_q.shape
      
        if not mat_q.is_contiguous():
            mat_q = mat_q.contiguous()

        if not mat_c.is_contiguous():
            mat_c = mat_c.contiguous()

        m = len(offsets)
        
        out = torch.zeros([n, h, d], dtype=mat_q.dtype, device=mat_q.device)
       
        _scatter_qc_fwd_kernel[(m, h)](
            mat_q, mat_c, offsets, counts, out, h=h, d=d, g=BLOCK_SIZE
        )
        
        ctx.save_for_backward(mat_q, mat_c, offsets, counts)
        
        ctx.d = d
        ctx.m = m
        ctx.h = h

        return out
    
    @staticmethod
    def backward(ctx, dout):
       
        mat_q, mat_c, offsets, counts = ctx.saved_tensors
        dq = torch.zeros_like(mat_q, dtype=torch.float32)
        dc = torch.zeros_like(mat_c, dtype=torch.float32)
      
        d = ctx.d
        m = ctx.m
        h = ctx.h

        _scatter_qc_bwd_kernel[(m,h)](
            dout.contiguous(), mat_q, mat_c, offsets, counts, dq, dc, h=h, d=d, g=BLOCK_SIZE
        )
       
        return dq, dc, None, None

scatter_matmul_qc = ScatterMatmulQC.apply          


def scatter_normalize(src, batch_win_ids):
    src_sum = torch_scatter.scatter_add(src ** 2, batch_win_ids, dim=0)
    src_sum = torch.sqrt(src_sum + 1e-6)
    return src / (src_sum[batch_win_ids, ...] + 1e-6)

class ScatterAttention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, indice_key=None):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads    
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # self.scale = self.head_dim ** -0.5
        self.pool = spconv.SparseSumPool2d(3, 1, 1, indice_key=indice_key)
      
    def forward(self, x, offsets, counts, batch_win_inds, 
                batch_win_coords, win_bev_shape, batch_size, indice_dict):
        
        N, C = x.shape
        
        qkv = self.qkv(x).reshape(N, 3, C)

        q, k, v = qkv.unbind(1)

        q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])

        q = F.relu(q)
        k = F.relu(k)

        kv = scatter_matmul_kv(k, v, offsets, counts)
        s = torch_scatter.scatter_add(k, batch_win_inds, dim=0)
        
        kv_tensor = spconv.SparseConvTensor(
            kv.view(-1, self.num_heads * self.head_dim * self.head_dim), 
            batch_win_coords, win_bev_shape, batch_size
        )
        if indice_dict is not None:
            kv_tensor.indice_dict = indice_dict
        kv = self.pool(kv_tensor)
        
        s_tensor = spconv.SparseConvTensor(
            s.view(-1, self.num_heads * self.head_dim),
            batch_win_coords, win_bev_shape, batch_size
        )
        s_tensor.indice_dict = kv.indice_dict
        s = self.pool(s_tensor)
     
        kv = kv.features.view(-1, self.num_heads, self.head_dim, self.head_dim)
        s = s.features.view(-1, self.num_heads, self.head_dim)

      
        y = scatter_matmul_qc(q, kv, offsets, counts)
        z = torch.sum(s[batch_win_inds, ...] * q, -1, keepdim=True)
       
        y = y / (z + 1e-6)

        y = rearrange(y, "n h c -> n (h c)", h=self.num_heads)
        
        y = self.proj(y)
       
        return y,  s_tensor.indice_dict



