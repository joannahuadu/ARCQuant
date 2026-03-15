import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import *
from optional_agemm import (
    AGEMM_IMPORT_ERROR,
    DEVICE_SM,
    HAS_AGEMM,
    require_agemm,
    warn_agemm_fallback_once,
)

import math
import random


def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def NVFP4_reorder_quantize_w(w, reorder_index, select_num):
    scale = torch.max(w.abs()).float() / (448.0*6.0)
    qw, scale_w = require_agemm().reorder_quantize_w(w / scale, reorder_index, select_num)
    return qw, scale_w, scale
    
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        select_num, 
        reorder_index,
        out_reorder_index=None,
        quant_type='NVFP4',
        *,
        reorder_xw: bool = True,
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
    
        
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
        self.select_num = select_num
        self.quant_type = quant_type

        self.use_agemm = bool(self.quant_type == "NVFP4" and HAS_AGEMM and reorder_xw)

        if self.use_agemm:
            # self.W, self.scale_w, self.scale = NVFP4_reorder_quantize_w((originalLayer.weight.data), torch.arange(self.in_features).to(torch.int16).cuda(), 0)
            W, scale_w, scale = NVFP4_reorder_quantize_w(
                (originalLayer.weight.data), reorder_index.to(torch.int16).cuda(), select_num
            )
        else:
            if self.quant_type == "NVFP4" and not HAS_AGEMM:
                reason = (
                    f"SM{DEVICE_SM} detected" if DEVICE_SM is not None else "no CUDA device detected"
                )
                if AGEMM_IMPORT_ERROR is not None:
                    reason = f"import failed: {AGEMM_IMPORT_ERROR}"
                warn_agemm_fallback_once(reason)

            if reorder_xw:
                w_reordered = torch.index_select(
                    originalLayer.weight.data, 1, reorder_index.to(torch.int32).cuda()
                )
                fake_reorder_index = torch.arange(self.in_features, device=w_reordered.device)
            else:
                w_reordered = originalLayer.weight.data
                fake_reorder_index = reorder_index.to(device=w_reordered.device, dtype=torch.long)
            W, scale_w, scale = fake_reorder_quantize_w(
                w_reordered,
                fake_reorder_index,
                select_num,
                dtype=self.quant_type,
            )
        
        self.register_buffer("W", W, persistent=False)
        self.register_buffer("scale_w", scale_w, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    def forward(self, x):
        qx, scale_x, scale, bsz, q_len = x

        if self.use_agemm:
            y = require_agemm().matmul(qx, self.W, scale_x, self.scale_w, scale * self.scale)
        else:
            y = F.linear(qx, self.W)
        
        if self.bias is not None:
            y = y + self.bias

        if bsz is not None:
            y = y.reshape(bsz, q_len, -1)
        else:
            y = y.reshape(q_len, -1)
        return y

    