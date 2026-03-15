import torch
import torch.nn.functional as F
import numpy as np
import gc

import math
import random


def quantize_e2m1(tensor):
    representable_vals = torch.tensor([
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    ], device=tensor.device, dtype=tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    return representable_vals[indices]

def dequantize_e2m1(tensor):
    return tensor

def quantize_int4(tensor):
    representable_vals = torch.tensor([
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    ], device=tensor.device, dtype=tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    return representable_vals[indices]

def dequantize_int4(tensor):
    return tensor

def quantize_ue4m3(tensor):
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2**exponent) - 1.0 
    
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    
    reconstructed_val = (1 + quantized_mantissa_val) * (2**exponent)
    return reconstructed_val

def dequantize_ue4m3(tensor):
    return tensor

def quantize_ue8m0(tensor):
    exponent = torch.ceil(torch.log2(tensor + 1e-9))
    exponent = torch.clamp(exponent, min=-127, max=127)
    
    reconstructed_val = (2**exponent)
    return reconstructed_val

def dequantize_ue8m0(tensor):
    return tensor


def quantize_s1p2(tensor):

    representable_vals = torch.tensor([
        -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
        0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    ], device=tensor.device, dtype=tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    return representable_vals[indices]

def quantize_e6m2(tensor):

    tensor = torch.clamp(tensor, min=2.0**(-48), max=2.0**15 * 1.5)
    
    exponent = torch.floor(torch.log2(tensor))
    mantissa_val = tensor / (2.0**exponent) - 1.0 
    
    quantized_mantissa_val = torch.round(mantissa_val * 4.0) / 4.0
    
    overflow = quantized_mantissa_val >= 1.0
    quantized_mantissa_val[overflow] = 0.0
    exponent[overflow] += 1.0
    
    exponent = torch.clamp(exponent, min=-48, max=15)
    
    is_nan = (exponent == 15) & (quantized_mantissa_val == 0.75)
    quantized_mantissa_val[is_nan] = 0.5
    
    reconstructed_val = (1.0 + quantized_mantissa_val) * (2.0**exponent)
    return reconstructed_val

def quantize_hif4_tensor(tensor, group_size=64):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    N = reshaped_tensor.shape[0]
    
    V16 = torch.max(torch.abs(reshaped_tensor.view(N, 16, 4)), dim=2)[0]
    V8 = torch.max(V16.view(N, 8, 2), dim=2)[0]
    Vmax = torch.max(V8, dim=1, keepdim=True)[0]
    
    SF_BF16 = Vmax / 7.0
    SF_BF16[SF_BF16 == 0] = 2.0**(-48)
    del Vmax  
    
    E6M2 = quantize_e6m2(SF_BF16)
    E6M2_REC = 1.0 / E6M2
    del SF_BF16 
    
    E1_8 = (V8 * E6M2_REC >= 4.0).float()
    del V8 
    
    E1_8_expanded = E1_8.repeat_interleave(2, dim=1) 
    E1_16 = (V16 * E6M2_REC * (2.0 ** (-E1_8_expanded)) >= 2.0).float()
    del V16, E1_8_expanded 
    
    E1_8_full = E1_8.repeat_interleave(8, dim=1)
    E1_16_full = E1_16.repeat_interleave(4, dim=1)
    
    V64_scaled = reshaped_tensor * E6M2_REC * (2.0 ** (-E1_8_full)) * (2.0 ** (-E1_16_full))
    
    S1P2_64 = quantize_s1p2(V64_scaled)
    del V64_scaled 
    
    dequantized_tensor_groups = S1P2_64 * E6M2 * (2.0 ** E1_8_full) * (2.0 ** E1_16_full)
    del S1P2_64, E1_8_full, E1_16_full, E6M2, E6M2_REC 
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)



def quantize_nvfp4_tensor(tensor, group_size=16):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 6.0
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue4m3(scale)
    dequantized_scale = dequantize_ue4m3(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e2m1_tensor = quantize_e2m1(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e2m1(quantized_e2m1_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

def quantize_mxfp4_tensor(tensor, group_size=32):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 6.0
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue8m0(scale)
    dequantized_scale = dequantize_ue8m0(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e2m1_tensor = quantize_e2m1(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e2m1(quantized_e2m1_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

def quantize_int4_tensor(tensor, group_size=128):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 7
    scale[scale == 0] = 1e-9 
    
    dequantized_scale = scale
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_int4_tensor = quantize_int4(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_int4(quantized_int4_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

def get_e3m2_values(device, dtype):
    vals =[0.0]
    vals.extend([0.0625, 0.125, 0.1875])
    
    mantissas =[1.0, 1.25, 1.5, 1.75]
    for E in range(1, 8): 
        exponent_val = 2 ** (E - 3)
        for m in mantissas:
            vals.append(m * exponent_val)
            
    pos_vals = torch.tensor(vals, device=device, dtype=dtype)
    all_vals = torch.cat([-pos_vals, pos_vals]).unique()
    return torch.sort(all_vals)[0]

def quantize_e3m2(tensor):
    representable_vals = get_e3m2_values(tensor.device, tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    
    return representable_vals[indices]

def dequantize_e3m2(tensor):
    return tensor

def quantize_mxfp6_tensor(tensor, group_size=32):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    
    scale = max_abs_val / 28.0 
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue8m0(scale)
    dequantized_scale = dequantize_ue8m0(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e3m2_tensor = quantize_e3m2(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e3m2(quantized_e3m2_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)


def fake_reorder_quantize_w(w, reorder_index, select_num, dtype='NVFP4'):
    orig_dtype = w.dtype 
    
    if dtype == "NVFP4":
        scale = torch.max(w.abs()).to(torch.float32) / (448.0*6.0)
        quantize_func = quantize_nvfp4_tensor
    elif dtype == "MXFP4":
        scale = torch.tensor(1.0, device=w.device, dtype=torch.float32)
        quantize_func = quantize_mxfp4_tensor
    elif dtype == "HiF4":
        scale = torch.tensor(1.0, device=w.device, dtype=torch.float32)
        quantize_func = quantize_hif4_tensor
    else:
        scale = torch.tensor(1.0, device=w.device, dtype=torch.float32)
        quantize_func = quantize_int4_tensor
    
    index = reorder_index.to(torch.int32)
    w = torch.index_select(w, 1, index)
    w_fp32 = w.to(torch.float32) / scale
    scale_w = w_fp32.abs().max(dim=1, keepdim=True)[0]
    
    if select_num == 0:
        q_w = quantize_func(w_fp32) * scale
        return q_w.to(orig_dtype), (scale_w * scale).to(orig_dtype), scale.to(orig_dtype)
    else:
        topk_index = torch.arange(w.shape[-1], device=w.device)[-select_num:]
        # q_w = quantize_func(w_fp32)
        # inv_index = torch.empty_like(index)
        # inv_index[index] = torch.arange(index.numel(), device=index.device).to(torch.int32)
        # q_w = torch.index_select(q_w, 1, inv_index)
        q_w = torch.cat([quantize_func(w_fp32), quantize_func(w_fp32[:, topk_index])], dim=1) * scale
        return q_w.to(orig_dtype), (scale_w * scale).to(orig_dtype), scale.to(orig_dtype)

def fake_reorder_quantize_x(x, reorder_index, select_num, dtype='NVFP4', *, ste: bool = False):
    orig_dtype = x.dtype  
    
    if dtype == "NVFP4":
        scale = torch.max(x.abs()).to(torch.float32) / (448.0*6.0)
        quantize_func = quantize_nvfp4_tensor
    elif dtype == "MXFP4":
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        quantize_func = quantize_mxfp4_tensor
    elif dtype == "HiF4":
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        quantize_func = quantize_hif4_tensor
    else:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        quantize_func = quantize_int4_tensor
    
    index = reorder_index.to(torch.int32)
    x = torch.index_select(x, 1, index)
    x_fp32 = x.to(torch.float32) / scale
    scale_x = x_fp32.abs().max(dim=1, keepdim=True)[0]
    
    if select_num == 0:
        q_x = quantize_func(x_fp32) * scale
        q_x = q_x.to(orig_dtype)
        if ste:
            q_x = x + (q_x - x).detach()
        return q_x, scale_x.to(orig_dtype), scale.to(orig_dtype)
    else:
        topk_index = torch.arange(x.shape[-1], device=x.device)[-select_num:]
        q_x = quantize_func(x_fp32)
        error_e = x_fp32 - q_x
        q_error_k = quantize_func(error_e[:, topk_index])
        # inv_index = torch.empty_like(index)
        # inv_index[index] = torch.arange(index.numel(), device=index.device).to(torch.int32)
        # q_x = torch.index_select(q_x, 1, inv_index)
        ret_x = torch.cat([q_x, q_error_k], dim=1) * scale
        ret_x = ret_x.to(orig_dtype)
        if ste:
            main_dim = int(x.shape[1])
            main = x + (ret_x[:, :main_dim] - x).detach()
            extra = ret_x[:, main_dim:].detach()
            ret_x = torch.cat([main, extra], dim=1)
        return ret_x, scale_x.to(orig_dtype), scale.to(orig_dtype)

@torch.no_grad()
def hadamard_transform(x, normalize=True, block_size=-1):
    n = x.shape[-1]
    if block_size == -1:
        if n <= 0 or (n & (n - 1)) != 0:
            return x
    else:
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size {block_size}")
        if n % block_size != 0:
            raise ValueError(f" {n} block_size {block_size}")
    
    original_shape = x.shape
    
    if block_size != -1:
        num_blocks = n // block_size
        x = x.view(-1, num_blocks, block_size)
        batch_dim = x.shape[0]
        current_n = block_size
    else:
        x = x.view(-1, n)
        batch_dim = x.shape[0]
        current_n = n
    
    h = x.clone()
    num_stages = int(torch.log2(torch.tensor(current_n, dtype=torch.float32)).item())
    
    for stage in range(num_stages):
        stage_block_size = 2 ** (stage + 1)
        half_block_size = stage_block_size // 2
        if block_size != -1:
            temp = h.view(batch_dim, -1, stage_block_size)
        else:
            temp = h.view(batch_dim, -1, stage_block_size)
        front_half = temp[:, :, :half_block_size]
        back_half = temp[:, :, half_block_size:]
        new_front = front_half + back_half
        new_back = front_half - back_half
        h = torch.cat([new_front, new_back], dim=2)
        if block_size != -1:
            h = h.view(batch_dim, -1, current_n)
        else:
            h = h.view(batch_dim, current_n)
    
    if normalize:
        h = h / torch.sqrt(torch.tensor(current_n, dtype=torch.float32))
    if block_size != -1:
        h = h.view(-1, num_blocks * block_size)
    h = h.view(original_shape)
    return h