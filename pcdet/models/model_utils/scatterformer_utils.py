import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch.nn as nn
from einops import rearrange
import torch_scatter

GROUP_SIZE=32

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
            mat_k, mat_v, offsets, counts, kv, h=h, d=d, g=GROUP_SIZE
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
            dout.contiguous(), mat_k, mat_v, offsets, counts, dk, dv, h=h, d=d, g=GROUP_SIZE
        )
       
        return dk, dv, None, None, None

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
            mat_q, mat_c, offsets, counts, out, h=h, d=d, g=GROUP_SIZE
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
            dout.contiguous(), mat_q, mat_c, offsets, counts, dq, dc, h=h, d=d, g=GROUP_SIZE
        )
       
        return dq, dc, None, None, None

scatter_matmul_qc = ScatterMatmulQC.apply        



def scatter_normalize(src, batch_win_ids):
    src_sum = torch_scatter.scatter_add(src ** 2, batch_win_ids, dim=0)
    src_sum = torch.sqrt(src_sum + 1e-16)
    return src / (src_sum[batch_win_ids, ...] + 1e-8)

class ScatterAttention(nn.Module):

    def __init__(self, dim, num_heads, attn_type='xcit'):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads    
        


        self.qkv = nn.Linear(dim, dim * 4)
        self.norm = nn.LayerNorm(self.head_dim)
       
        self.proj = nn.Linear(dim, dim, bias=False)
        # self.scaling = self.head_dim ** -0.5
        
        
    def forward(self, x, offsets, counts, batch_win_inds, pe=None):
        
        N, C = x.shape
         
        qkv = self.qkv(x).reshape(N, 4, C)
        q, k, v, g = qkv.unbind(1)
        if pe is not None:
            q = q + pe
            k = k + pe
        q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])
        
        s = scatter_matmul_kv(k, v, offsets, counts)
        o = scatter_matmul_qc(q, s, offsets, counts)
        
        o = rearrange(self.norm(o), "n h c -> n (h c)", h=self.num_heads)
        
        o = F.silu(g) * o
        
        return self.proj(o)
       
       





