from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PROB_LT_THRESHOLDS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
PROB_GT_THRESHOLDS = (0.9, 0.99, 0.999)
ROW_MAX_GT_THRESHOLDS = (0.9, 0.99, 0.999)
NORM_ENTROPY_LT_THRESHOLDS = (0.1, 0.01)
TOP1_LAG_LE_THRESHOLDS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@dataclass
class SoftmaxStatsConfig:
    sample_per_call: int = 262_144
    max_calls: int = 0  # 0 = unlimited
    bins_linear: int = 200
    bins_log10: int = 240
    log10_min: float = -12.0
    eps: float = 1e-12
    only_ndim: int = 4
    only_last_dim_min: int = 16
    entropy_rows_per_call: int = 8_192
    entropy_bins: int = 200
    per_layer: bool = False
    per_head: bool = False
    head_dim: int = 1
    row_std_rows_per_call: int = 2_048
    top1_lag_rows_per_call: int = 2_048
    top1_lag_bins: int = 512
    top1_lag_max: int = 8_192


class SoftmaxStatsCollector:
    def __init__(self, config: SoftmaxStatsConfig, *, logger=None) -> None:
        self.config = config
        self.logger = logger

        self.n_calls: int = 0
        self.n_skipped: int = 0
        self.n_sampled: int = 0
        self.n_zero: int = 0

        self._prob_lt_counts = [0] * len(PROB_LT_THRESHOLDS)
        self._prob_gt_counts = [0] * len(PROB_GT_THRESHOLDS)

        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.min_val: float = math.inf
        self.max_val: float = -math.inf

        self.linear_counts = torch.zeros(self.config.bins_linear, dtype=torch.int64)
        self.log10_counts = torch.zeros(self.config.bins_log10, dtype=torch.int64)
        self.max_linear_counts = torch.zeros(self.config.bins_linear, dtype=torch.int64)

        self.row_max_n: int = 0
        self.row_max_sum: float = 0.0
        self.row_max_sumsq: float = 0.0
        self.row_max_min: float = math.inf
        self.row_max_max: float = -math.inf
        self._row_max_gt_counts = [0] * len(ROW_MAX_GT_THRESHOLDS)

        self.entropy_n: int = 0
        self.entropy_sum: float = 0.0
        self.entropy_sumsq: float = 0.0
        self.entropy_min: float = math.inf
        self.entropy_max: float = -math.inf

        self.entropy_norm_n: int = 0
        self.entropy_norm_sum: float = 0.0
        self.entropy_norm_sumsq: float = 0.0
        self.entropy_norm_min: float = math.inf
        self.entropy_norm_max: float = -math.inf
        self.entropy_norm_counts = torch.zeros(self.config.entropy_bins, dtype=torch.int64)
        self._entropy_norm_lt_counts = [0] * len(NORM_ENTROPY_LT_THRESHOLDS)

        self.top1_lag_n: int = 0
        self.top1_lag_sum: float = 0.0
        self.top1_lag_sumsq: float = 0.0
        self.top1_lag_min: float = math.inf
        self.top1_lag_max: float = -math.inf
        self.top1_lag_counts = torch.zeros(self.config.top1_lag_bins, dtype=torch.int64)
        self.top1_lag_neg_n: int = 0
        self.top1_lag_overflow_n: int = 0
        self._top1_lag_le_counts = [0] * len(TOP1_LAG_LE_THRESHOLDS)

        self._current_layer: int | None = None
        self._layer_stack: list[int | None] = []

        self.entropy_norm_by_layer_n: dict[int, int] = {}
        self.entropy_norm_by_layer_sum: dict[int, float] = {}
        self.entropy_norm_by_layer_sumsq: dict[int, float] = {}
        self.entropy_norm_by_layer_min: dict[int, float] = {}
        self.entropy_norm_by_layer_max: dict[int, float] = {}
        self.entropy_norm_counts_by_layer: dict[int, torch.Tensor] = {}
        self.entropy_norm_lt_counts_by_layer: dict[int, list[int]] = {}

        self.top1_lag_by_layer_n: dict[int, int] = {}
        self.top1_lag_by_layer_sum: dict[int, float] = {}
        self.top1_lag_by_layer_sumsq: dict[int, float] = {}
        self.top1_lag_by_layer_min: dict[int, float] = {}
        self.top1_lag_by_layer_max: dict[int, float] = {}
        self.top1_lag_counts_by_layer: dict[int, torch.Tensor] = {}
        self.top1_lag_neg_by_layer_n: dict[int, int] = {}
        self.top1_lag_overflow_by_layer_n: dict[int, int] = {}
        self.top1_lag_le_counts_by_layer: dict[int, list[int]] = {}

        self.row_std_by_layer_head_n: dict[int, list[int]] = {}
        self.row_std_by_layer_head_sum: dict[int, list[float]] = {}
        self.row_std_by_layer_head_sumsq: dict[int, list[float]] = {}
        self.row_std_by_layer_head_min: dict[int, list[float]] = {}
        self.row_std_by_layer_head_max: dict[int, list[float]] = {}

        self._orig_torch_softmax = None
        self._orig_tensor_softmax = None
        self._top1_lag_le_thr_t = None

    def _infer_q_dim(self, out: torch.Tensor) -> int | None:
        if not torch.is_tensor(out) or out.dim() != 4:
            return None
        head_dim = int(self.config.head_dim) % int(out.dim())
        if head_dim == out.dim() - 1:
            head_dim = -1
        # Assume common layouts:
        # - [B, H, Q, K] when head_dim=1 -> q_dim=2
        # - [B, Q, H, K] when head_dim=2 -> q_dim=1
        if head_dim == 2:
            return 1
        return 2

    def push_layer(self, layer_idx: int) -> None:
        self._layer_stack.append(self._current_layer)
        self._current_layer = int(layer_idx)

    def pop_layer(self) -> None:
        self._current_layer = self._layer_stack.pop() if self._layer_stack else None

    def _should_record(self, out: torch.Tensor, dim: int) -> bool:
        if not torch.is_tensor(out):
            return False
        if out.dim() != self.config.only_ndim:
            return False
        if (dim % out.dim()) != (out.dim() - 1):
            return False
        if out.shape[-1] < self.config.only_last_dim_min:
            return False
        if self.config.max_calls > 0 and self.n_calls >= self.config.max_calls:
            return False
        return True

    def _update_entropy_norm_by_layer(self, layer: int, ent_norm: torch.Tensor) -> None:
        if ent_norm.numel() == 0:
            return

        layer = int(layer)
        n = int(ent_norm.numel())
        s = float(ent_norm.sum().item())
        ss = float((ent_norm * ent_norm).sum().item())
        mn = float(ent_norm.min().item())
        mx = float(ent_norm.max().item())

        self.entropy_norm_by_layer_n[layer] = self.entropy_norm_by_layer_n.get(layer, 0) + n
        self.entropy_norm_by_layer_sum[layer] = self.entropy_norm_by_layer_sum.get(layer, 0.0) + s
        self.entropy_norm_by_layer_sumsq[layer] = self.entropy_norm_by_layer_sumsq.get(layer, 0.0) + ss
        self.entropy_norm_by_layer_min[layer] = min(self.entropy_norm_by_layer_min.get(layer, math.inf), mn)
        self.entropy_norm_by_layer_max[layer] = max(self.entropy_norm_by_layer_max.get(layer, -math.inf), mx)

        hist = torch.histc(ent_norm, bins=self.config.entropy_bins, min=0.0, max=1.0).to(dtype=torch.int64).cpu()
        if layer in self.entropy_norm_counts_by_layer:
            self.entropy_norm_counts_by_layer[layer] += hist
        else:
            self.entropy_norm_counts_by_layer[layer] = hist

        lt_counts = self.entropy_norm_lt_counts_by_layer.get(layer)
        if lt_counts is None:
            lt_counts = [0] * len(NORM_ENTROPY_LT_THRESHOLDS)
            self.entropy_norm_lt_counts_by_layer[layer] = lt_counts
        for i, thr in enumerate(NORM_ENTROPY_LT_THRESHOLDS):
            lt_counts[i] += int((ent_norm < thr).sum().item())

    def _update_top1_lag_by_layer(
        self,
        layer: int,
        lag_pos: torch.Tensor,
        *,
        neg_n: int,
        overflow_n: int,
        le_counts_in: list[int] | None = None,
        hist_cpu: torch.Tensor | None = None,
    ) -> None:
        if lag_pos.numel() == 0:
            return

        layer = int(layer)
        lag_f = lag_pos.to(dtype=torch.float32)

        n = int(lag_f.numel())
        s = float(lag_f.sum().item())
        ss = float((lag_f * lag_f).sum().item())
        mn = float(lag_f.min().item())
        mx = float(lag_f.max().item())

        self.top1_lag_by_layer_n[layer] = self.top1_lag_by_layer_n.get(layer, 0) + n
        self.top1_lag_by_layer_sum[layer] = self.top1_lag_by_layer_sum.get(layer, 0.0) + s
        self.top1_lag_by_layer_sumsq[layer] = self.top1_lag_by_layer_sumsq.get(layer, 0.0) + ss
        self.top1_lag_by_layer_min[layer] = min(self.top1_lag_by_layer_min.get(layer, math.inf), mn)
        self.top1_lag_by_layer_max[layer] = max(self.top1_lag_by_layer_max.get(layer, -math.inf), mx)

        self.top1_lag_neg_by_layer_n[layer] = self.top1_lag_neg_by_layer_n.get(layer, 0) + int(neg_n)
        self.top1_lag_overflow_by_layer_n[layer] = self.top1_lag_overflow_by_layer_n.get(layer, 0) + int(
            overflow_n
        )

        if hist_cpu is None:
            lag_clip = lag_f.clamp_max(float(self.config.top1_lag_max))
            hist_cpu = torch.histc(
                lag_clip,
                bins=int(self.config.top1_lag_bins),
                min=0.0,
                max=float(self.config.top1_lag_max),
            ).to(dtype=torch.int64, device="cpu")
        if layer in self.top1_lag_counts_by_layer:
            self.top1_lag_counts_by_layer[layer] += hist_cpu
        else:
            self.top1_lag_counts_by_layer[layer] = hist_cpu

        le_counts_buf = self.top1_lag_le_counts_by_layer.get(layer)
        if le_counts_buf is None:
            le_counts_buf = [0] * len(TOP1_LAG_LE_THRESHOLDS)
            self.top1_lag_le_counts_by_layer[layer] = le_counts_buf
        if le_counts_in is None:
            lag_i = lag_pos.to(dtype=torch.int64)
            if self._top1_lag_le_thr_t is None or self._top1_lag_le_thr_t.device != lag_i.device:
                self._top1_lag_le_thr_t = torch.tensor(
                    TOP1_LAG_LE_THRESHOLDS, device=lag_i.device, dtype=lag_i.dtype
                )
            c = (lag_i.unsqueeze(-1) <= self._top1_lag_le_thr_t).sum(dim=0).to(dtype=torch.int64, device="cpu")
            le_counts_in = c.tolist()
        for i, v in enumerate(le_counts_in):
            le_counts_buf[i] += int(v)

    def _ensure_row_std_buffers(self, layer: int, n_heads: int) -> None:
        layer = int(layer)
        n_heads = int(n_heads)
        if n_heads <= 0:
            return
        if layer not in self.row_std_by_layer_head_n:
            self.row_std_by_layer_head_n[layer] = [0] * n_heads
            self.row_std_by_layer_head_sum[layer] = [0.0] * n_heads
            self.row_std_by_layer_head_sumsq[layer] = [0.0] * n_heads
            self.row_std_by_layer_head_min[layer] = [math.inf] * n_heads
            self.row_std_by_layer_head_max[layer] = [-math.inf] * n_heads
            return

        cur = len(self.row_std_by_layer_head_n[layer])
        if n_heads > cur:
            extra = n_heads - cur
            self.row_std_by_layer_head_n[layer].extend([0] * extra)
            self.row_std_by_layer_head_sum[layer].extend([0.0] * extra)
            self.row_std_by_layer_head_sumsq[layer].extend([0.0] * extra)
            self.row_std_by_layer_head_min[layer].extend([math.inf] * extra)
            self.row_std_by_layer_head_max[layer].extend([-math.inf] * extra)

    @torch.no_grad()
    def observe(self, out: torch.Tensor, *, dim: int) -> None:
        if not self._should_record(out, dim):
            self.n_skipped += 1
            return

        self.n_calls += 1

        flat = out.detach().reshape(-1)
        if flat.numel() == 0:
            return

        sample_n = min(int(self.config.sample_per_call), int(flat.numel()))
        if sample_n < flat.numel():
            idx = torch.randint(0, flat.numel(), (sample_n,), device=flat.device)
            vals = flat[idx]
        else:
            vals = flat

        vals = vals.float()
        if vals.numel() == 0:
            return

        finite = torch.isfinite(vals)
        if not finite.all():
            vals = vals[finite]
        if vals.numel() == 0:
            return

        self.n_sampled += int(vals.numel())
        self.n_zero += int((vals == 0).sum().item())

        vmin = float(vals.min().item())
        vmax = float(vals.max().item())
        self.min_val = min(self.min_val, vmin)
        self.max_val = max(self.max_val, vmax)

        vsum = float(vals.sum().item())
        vsumsq = float((vals * vals).sum().item())
        self.sum += vsum
        self.sumsq += vsumsq

        # Value histogram in [0, 1]
        vals01 = vals.clamp_(0.0, 1.0)
        hist = torch.histc(vals01, bins=self.config.bins_linear, min=0.0, max=1.0)
        self.linear_counts += hist.to(dtype=torch.int64).cpu()

        for i, thr in enumerate(PROB_LT_THRESHOLDS):
            self._prob_lt_counts[i] += int((vals01 < thr).sum().item())
        for i, thr in enumerate(PROB_GT_THRESHOLDS):
            self._prob_gt_counts[i] += int((vals01 > thr).sum().item())

        # log10 histogram in [log10_min, 0]
        logv = torch.log10(vals01.clamp_min(self.config.eps))
        logv = logv.clamp_(self.config.log10_min, 0.0)
        lhist = torch.histc(logv, bins=self.config.bins_log10, min=self.config.log10_min, max=0.0)
        self.log10_counts += lhist.to(dtype=torch.int64).cpu()

        # Row-wise max (sharpness proxy), still in [0, 1]
        row_max = out.detach().amax(dim=-1).reshape(-1).float().clamp_(0.0, 1.0)
        if row_max.numel():
            mhist = torch.histc(row_max, bins=self.config.bins_linear, min=0.0, max=1.0)
            self.max_linear_counts += mhist.to(dtype=torch.int64).cpu()
            self.row_max_n += int(row_max.numel())
            self.row_max_sum += float(row_max.sum().item())
            self.row_max_sumsq += float((row_max * row_max).sum().item())
            self.row_max_min = min(self.row_max_min, float(row_max.min().item()))
            self.row_max_max = max(self.row_max_max, float(row_max.max().item()))
            for i, thr in enumerate(ROW_MAX_GT_THRESHOLDS):
                self._row_max_gt_counts[i] += int((row_max > thr).sum().item())

        need_entropy = self.config.entropy_rows_per_call > 0 and out.shape[-1] > 1
        need_row_std = (
            self.config.per_head
            and self.config.row_std_rows_per_call > 0
            and self._current_layer is not None
            and out.dim() == 4
            and out.shape[-1] > 1
        )
        need_top1_lag = (
            self.config.top1_lag_rows_per_call > 0
            and self.config.top1_lag_bins > 0
            and self.config.top1_lag_max > 0
            and out.dim() == 4
            and out.shape[-1] > 1
        )

        # Per-row entropy (normalized) and/or per-layer-per-head row-wise std.
        if need_entropy or need_row_std or need_top1_lag:
            rows = out.detach().reshape(-1, out.shape[-1])
            if rows.numel():
                total_rows = int(rows.shape[0])
                max_rows = 0
                if need_entropy:
                    max_rows = max(max_rows, int(self.config.entropy_rows_per_call))
                if need_row_std:
                    max_rows = max(max_rows, int(self.config.row_std_rows_per_call))
                if need_top1_lag:
                    max_rows = max(max_rows, int(self.config.top1_lag_rows_per_call))
                max_rows = min(max_rows, total_rows)
                if max_rows < total_rows:
                    ridx_all = torch.randint(0, total_rows, (max_rows,), device=rows.device)
                    rows_all = rows[ridx_all]
                else:
                    ridx_all = None
                    rows_all = rows

                if need_entropy:
                    n_ent = min(int(self.config.entropy_rows_per_call), int(rows_all.shape[0]))
                    rows_s = rows_all[:n_ent]
                    p = rows_s.float().clamp_min(0.0)
                    ent = -(p * (p + self.config.eps).log()).sum(dim=-1)
                    ent = ent[torch.isfinite(ent)]
                    if ent.numel():
                        self.entropy_n += int(ent.numel())
                        self.entropy_sum += float(ent.sum().item())
                        self.entropy_sumsq += float((ent * ent).sum().item())
                        self.entropy_min = min(self.entropy_min, float(ent.min().item()))
                        self.entropy_max = max(self.entropy_max, float(ent.max().item()))

                        denom = math.log(int(out.shape[-1]))
                        if denom > 0:
                            ent_norm = (ent / denom).clamp_(0.0, 1.0)
                            self.entropy_norm_n += int(ent_norm.numel())
                            self.entropy_norm_sum += float(ent_norm.sum().item())
                            self.entropy_norm_sumsq += float((ent_norm * ent_norm).sum().item())
                            self.entropy_norm_min = min(self.entropy_norm_min, float(ent_norm.min().item()))
                            self.entropy_norm_max = max(self.entropy_norm_max, float(ent_norm.max().item()))
                            ehist = torch.histc(ent_norm, bins=self.config.entropy_bins, min=0.0, max=1.0)
                            self.entropy_norm_counts += ehist.to(dtype=torch.int64).cpu()
                            for i, thr in enumerate(NORM_ENTROPY_LT_THRESHOLDS):
                                self._entropy_norm_lt_counts[i] += int((ent_norm < thr).sum().item())
                            if self.config.per_layer and self._current_layer is not None:
                                self._update_entropy_norm_by_layer(self._current_layer, ent_norm)

                if need_top1_lag:
                    q_dim = self._infer_q_dim(out)
                    if q_dim is not None:
                        n_lag = min(int(self.config.top1_lag_rows_per_call), int(rows_all.shape[0]))
                        if n_lag > 0:
                            rows_l = rows_all[:n_lag]
                            if ridx_all is not None:
                                ridx_l = ridx_all[:n_lag]
                            else:
                                ridx_l = torch.arange(n_lag, device=rows.device)

                            top1_idx = rows_l.argmax(dim=-1).to(dtype=torch.int64)

                            d0, d1, d2, k_len = (
                                int(out.shape[0]),
                                int(out.shape[1]),
                                int(out.shape[2]),
                                int(out.shape[-1]),
                            )
                            denom12 = max(d1 * d2, 1)
                            i0 = ridx_l // denom12
                            rem = ridx_l - i0 * denom12
                            i1 = rem // max(d2, 1)
                            i2 = rem % max(d2, 1)
                            if int(q_dim) == 0:
                                q_idx = i0
                            elif int(q_dim) == 1:
                                q_idx = i1
                            else:
                                q_idx = i2

                            q_len = int(out.shape[int(q_dim)])
                            if q_len > 0 and k_len > 0:
                                offset = max(k_len - q_len, 0)
                                query_pos = q_idx.to(dtype=torch.int64) + int(offset)
                                lag = query_pos - top1_idx
                                neg_n = int((lag < 0).sum().item())
                                self.top1_lag_neg_n += neg_n
                                lag_pos = lag.clamp_min(0)
                                overflow_n = int((lag_pos > int(self.config.top1_lag_max)).sum().item())
                                self.top1_lag_overflow_n += overflow_n

                                lag_i = lag_pos.to(dtype=torch.int64)
                                if self._top1_lag_le_thr_t is None or self._top1_lag_le_thr_t.device != lag_i.device:
                                    self._top1_lag_le_thr_t = torch.tensor(
                                        TOP1_LAG_LE_THRESHOLDS, device=lag_i.device, dtype=lag_i.dtype
                                    )
                                c = (lag_i.unsqueeze(-1) <= self._top1_lag_le_thr_t).sum(dim=0).to(
                                    dtype=torch.int64, device="cpu"
                                )
                                le_counts = c.tolist()
                                for i, v in enumerate(le_counts):
                                    self._top1_lag_le_counts[i] += int(v)

                                lag_f = lag_pos.to(dtype=torch.float32)
                                if lag_f.numel():
                                    self.top1_lag_n += int(lag_f.numel())
                                    self.top1_lag_sum += float(lag_f.sum().item())
                                    self.top1_lag_sumsq += float((lag_f * lag_f).sum().item())
                                    self.top1_lag_min = min(self.top1_lag_min, float(lag_f.min().item()))
                                    self.top1_lag_max = max(self.top1_lag_max, float(lag_f.max().item()))

                                    lag_clip = lag_f.clamp_max(float(self.config.top1_lag_max))
                                    hist = torch.histc(
                                        lag_clip,
                                        bins=int(self.config.top1_lag_bins),
                                        min=0.0,
                                        max=float(self.config.top1_lag_max),
                                    )
                                    hist_cpu = hist.to(dtype=torch.int64).cpu()
                                    self.top1_lag_counts += hist_cpu
                                    if self.config.per_layer and self._current_layer is not None:
                                        self._update_top1_lag_by_layer(
                                            self._current_layer,
                                            lag_pos,
                                            neg_n=neg_n,
                                            overflow_n=overflow_n,
                                            le_counts_in=le_counts,
                                            hist_cpu=hist_cpu,
                                        )

                if need_row_std:
                    n_rs = min(int(self.config.row_std_rows_per_call), int(rows_all.shape[0]))
                    rows_r = rows_all[:n_rs]
                    p_r = rows_r.float().clamp_min(0.0)
                    row_std = p_r.std(dim=-1, unbiased=False)
                    finite = torch.isfinite(row_std)
                    if not finite.all():
                        row_std = row_std[finite]
                        if ridx_all is not None:
                            ridx_r = ridx_all[:n_rs][finite]
                        else:
                            ridx_r = torch.arange(n_rs, device=rows.device)[finite]
                    else:
                        ridx_r = ridx_all[:n_rs] if ridx_all is not None else torch.arange(n_rs, device=rows.device)

                    if row_std.numel():
                        head_dim = int(self.config.head_dim) % int(out.dim())
                        if head_dim == out.dim() - 1:
                            head_dim = -1
                        if head_dim in (0, 1, 2):
                            d0, d1, d2 = (int(out.shape[0]), int(out.shape[1]), int(out.shape[2]))
                            if head_dim == 0:
                                denom = d1 * d2
                                head_ids = ridx_r // max(denom, 1)
                                n_heads = d0
                            elif head_dim == 1:
                                head_ids = (ridx_r // max(d2, 1)) % max(d1, 1)
                                n_heads = d1
                            else:
                                head_ids = ridx_r % max(d2, 1)
                                n_heads = d2

                            if n_heads > 0:
                                head_ids_c = head_ids.to(device="cpu", dtype=torch.int64)
                                row_std_c = row_std.to(device="cpu", dtype=torch.float32)
                                n_per_head = torch.bincount(head_ids_c, minlength=int(n_heads))
                                sum_per_head = torch.bincount(head_ids_c, weights=row_std_c, minlength=int(n_heads))
                                sumsq_per_head = torch.bincount(
                                    head_ids_c, weights=row_std_c * row_std_c, minlength=int(n_heads)
                                )

                                layer = int(self._current_layer)
                                self._ensure_row_std_buffers(layer, int(n_heads))
                                n_buf = self.row_std_by_layer_head_n[layer]
                                sum_buf = self.row_std_by_layer_head_sum[layer]
                                sumsq_buf = self.row_std_by_layer_head_sumsq[layer]
                                min_buf = self.row_std_by_layer_head_min[layer]
                                max_buf = self.row_std_by_layer_head_max[layer]
                                for h in range(int(n_heads)):
                                    nh = int(n_per_head[h].item())
                                    if nh <= 0:
                                        continue
                                    n_buf[h] += nh
                                    sum_buf[h] += float(sum_per_head[h].item())
                                    sumsq_buf[h] += float(sumsq_per_head[h].item())
                                    vals_h = row_std_c[head_ids_c == h]
                                    if vals_h.numel():
                                        min_buf[h] = min(min_buf[h], float(vals_h.min().item()))
                                        max_buf[h] = max(max_buf[h], float(vals_h.max().item()))

        if self.logger and (self.n_calls == 1 or (self.n_calls % 200 == 0)):
            row_max_mean = self.row_max_sum / max(self.row_max_n, 1)
            row_max_gt_099 = (
                self._row_max_gt_counts[1] / max(self.row_max_n, 1)
                if len(self._row_max_gt_counts) > 1
                else float("nan")
            )
            h_norm_mean = (
                self.entropy_norm_sum / max(self.entropy_norm_n, 1) if self.entropy_norm_n else float("nan")
            )
            h_norm_lt_001 = (
                self._entropy_norm_lt_counts[1] / max(self.entropy_norm_n, 1)
                if self.entropy_norm_n and len(self._entropy_norm_lt_counts) > 1
                else float("nan")
            )
            self.logger.info(
                f"[softmax_stats] calls={self.n_calls} sampled={self.n_sampled} "
                f"zero_ratio={(self.n_zero / max(self.n_sampled, 1)):.6f} "
                f"row_max_mean={row_max_mean:.6f} row_max_gt0.99={row_max_gt_099:.6f} "
                f"h_norm_mean={h_norm_mean:.6f} h_norm_lt0.01={h_norm_lt_001:.6f}"
            )

    def summary(self) -> dict[str, Any]:
        n = max(self.n_sampled, 1)
        mean = self.sum / n
        var = max(self.sumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        row_n = max(self.row_max_n, 1)
        row_max_mean = self.row_max_sum / row_n
        row_max_var = max(self.row_max_sumsq / row_n - row_max_mean * row_max_mean, 0.0)
        row_max_std = math.sqrt(row_max_var)

        ent_n = max(self.entropy_n, 1)
        ent_mean = self.entropy_sum / ent_n
        ent_var = max(self.entropy_sumsq / ent_n - ent_mean * ent_mean, 0.0)
        ent_std = math.sqrt(ent_var)

        entn_n = max(self.entropy_norm_n, 1)
        entn_mean = self.entropy_norm_sum / entn_n
        entn_var = max(self.entropy_norm_sumsq / entn_n - entn_mean * entn_mean, 0.0)
        entn_std = math.sqrt(entn_var)

        lag_n = max(self.top1_lag_n, 1)
        lag_mean = self.top1_lag_sum / lag_n
        lag_var = max(self.top1_lag_sumsq / lag_n - lag_mean * lag_mean, 0.0)
        lag_std = math.sqrt(lag_var)

        return {
            "n_calls": self.n_calls,
            "n_skipped": self.n_skipped,
            "n_sampled": self.n_sampled,
            "zero_ratio": self.n_zero / n,
            "min": None if not math.isfinite(self.min_val) else self.min_val,
            "max": None if not math.isfinite(self.max_val) else self.max_val,
            "mean": mean,
            "std": std,
            "prob_lt_thresholds": list(PROB_LT_THRESHOLDS),
            "prob_lt_ratio": [c / n for c in self._prob_lt_counts],
            "prob_gt_thresholds": list(PROB_GT_THRESHOLDS),
            "prob_gt_ratio": [c / n for c in self._prob_gt_counts],
            "row_max_n": self.row_max_n,
            "row_max_min": None if not math.isfinite(self.row_max_min) else self.row_max_min,
            "row_max_max": None if not math.isfinite(self.row_max_max) else self.row_max_max,
            "row_max_mean": row_max_mean,
            "row_max_std": row_max_std,
            "row_max_gt_thresholds": list(ROW_MAX_GT_THRESHOLDS),
            "row_max_gt_ratio": [c / row_n for c in self._row_max_gt_counts],
            "entropy_n": self.entropy_n,
            "entropy_min": None if not math.isfinite(self.entropy_min) else self.entropy_min,
            "entropy_max": None if not math.isfinite(self.entropy_max) else self.entropy_max,
            "entropy_mean": ent_mean if self.entropy_n else None,
            "entropy_std": ent_std if self.entropy_n else None,
            "entropy_norm_n": self.entropy_norm_n,
            "entropy_norm_min": None if not math.isfinite(self.entropy_norm_min) else self.entropy_norm_min,
            "entropy_norm_max": None if not math.isfinite(self.entropy_norm_max) else self.entropy_norm_max,
            "entropy_norm_mean": entn_mean if self.entropy_norm_n else None,
            "entropy_norm_std": entn_std if self.entropy_norm_n else None,
            "entropy_norm_lt_thresholds": list(NORM_ENTROPY_LT_THRESHOLDS),
            "entropy_norm_lt_ratio": [
                c / max(self.entropy_norm_n, 1) for c in self._entropy_norm_lt_counts
            ],
            "top1_lag_n": self.top1_lag_n,
            "top1_lag_min": None if not math.isfinite(self.top1_lag_min) else self.top1_lag_min,
            "top1_lag_max": None if not math.isfinite(self.top1_lag_max) else self.top1_lag_max,
            "top1_lag_mean": lag_mean if self.top1_lag_n else None,
            "top1_lag_std": lag_std if self.top1_lag_n else None,
            "top1_lag_neg_ratio": self.top1_lag_neg_n / max(self.top1_lag_n, 1),
            "top1_lag_overflow_ratio": self.top1_lag_overflow_n / max(self.top1_lag_n, 1),
            "top1_lag_le_thresholds": list(TOP1_LAG_LE_THRESHOLDS),
            "top1_lag_le_ratio": [c / max(self.top1_lag_n, 1) for c in self._top1_lag_le_counts],
            "config": {
                "sample_per_call": self.config.sample_per_call,
                "max_calls": self.config.max_calls,
                "bins_linear": self.config.bins_linear,
                "bins_log10": self.config.bins_log10,
                "log10_min": self.config.log10_min,
                "eps": self.config.eps,
                "only_ndim": self.config.only_ndim,
                "only_last_dim_min": self.config.only_last_dim_min,
                "entropy_rows_per_call": self.config.entropy_rows_per_call,
                "entropy_bins": self.config.entropy_bins,
                "per_layer": self.config.per_layer,
                "per_head": self.config.per_head,
                "head_dim": self.config.head_dim,
                "row_std_rows_per_call": self.config.row_std_rows_per_call,
                "top1_lag_rows_per_call": self.config.top1_lag_rows_per_call,
                "top1_lag_bins": self.config.top1_lag_bins,
                "top1_lag_max": self.config.top1_lag_max,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        entropy_norm_by_layer_summary: dict[str, Any] = {}
        if self.config.per_layer and self.entropy_norm_by_layer_n:
            for layer in sorted(self.entropy_norm_by_layer_n):
                n = max(int(self.entropy_norm_by_layer_n[layer]), 1)
                mean = float(self.entropy_norm_by_layer_sum[layer]) / n
                var = max(float(self.entropy_norm_by_layer_sumsq[layer]) / n - mean * mean, 0.0)
                std = math.sqrt(var)
                lt_counts = self.entropy_norm_lt_counts_by_layer.get(layer) or [0] * len(NORM_ENTROPY_LT_THRESHOLDS)
                entropy_norm_by_layer_summary[str(layer)] = {
                    "n": int(self.entropy_norm_by_layer_n[layer]),
                    "min": None
                    if not math.isfinite(self.entropy_norm_by_layer_min.get(layer, math.inf))
                    else float(self.entropy_norm_by_layer_min[layer]),
                    "max": None
                    if not math.isfinite(self.entropy_norm_by_layer_max.get(layer, -math.inf))
                    else float(self.entropy_norm_by_layer_max[layer]),
                    "mean": mean,
                    "std": std,
                    "lt_thresholds": list(NORM_ENTROPY_LT_THRESHOLDS),
                    "lt_ratio": [c / n for c in lt_counts],
                }
        top1_lag_by_layer_summary: dict[str, Any] = {}
        if self.config.per_layer and self.top1_lag_by_layer_n:
            for layer in sorted(self.top1_lag_by_layer_n):
                n = max(int(self.top1_lag_by_layer_n[layer]), 1)
                mean = float(self.top1_lag_by_layer_sum[layer]) / n
                var = max(float(self.top1_lag_by_layer_sumsq[layer]) / n - mean * mean, 0.0)
                std = math.sqrt(var)
                le_counts = self.top1_lag_le_counts_by_layer.get(layer) or [0] * len(TOP1_LAG_LE_THRESHOLDS)
                top1_lag_by_layer_summary[str(layer)] = {
                    "n": int(self.top1_lag_by_layer_n[layer]),
                    "min": None
                    if not math.isfinite(self.top1_lag_by_layer_min.get(layer, math.inf))
                    else float(self.top1_lag_by_layer_min[layer]),
                    "max": None
                    if not math.isfinite(self.top1_lag_by_layer_max.get(layer, -math.inf))
                    else float(self.top1_lag_by_layer_max[layer]),
                    "mean": mean,
                    "std": std,
                    "neg_ratio": self.top1_lag_neg_by_layer_n.get(layer, 0) / n,
                    "overflow_ratio": self.top1_lag_overflow_by_layer_n.get(layer, 0) / n,
                    "le_thresholds": list(TOP1_LAG_LE_THRESHOLDS),
                    "le_ratio": [c / n for c in le_counts],
                }
        row_std_by_layer_head_summary: dict[str, Any] = {}
        if self.config.per_head and self.row_std_by_layer_head_n:
            for layer in sorted(self.row_std_by_layer_head_n):
                n_list = self.row_std_by_layer_head_n[layer]
                sum_list = self.row_std_by_layer_head_sum[layer]
                sumsq_list = self.row_std_by_layer_head_sumsq[layer]
                min_list = self.row_std_by_layer_head_min[layer]
                max_list = self.row_std_by_layer_head_max[layer]
                heads: dict[str, Any] = {}
                for h in range(len(n_list)):
                    n = int(n_list[h])
                    if n <= 0:
                        continue
                    mean = float(sum_list[h]) / n
                    var = max(float(sumsq_list[h]) / n - mean * mean, 0.0)
                    std = math.sqrt(var)
                    heads[str(h)] = {
                        "n": n,
                        "min": None if not math.isfinite(min_list[h]) else float(min_list[h]),
                        "max": None if not math.isfinite(max_list[h]) else float(max_list[h]),
                        "mean": mean,
                        "std": std,
                    }
                if heads:
                    row_std_by_layer_head_summary[str(layer)] = heads
        return {
            "summary": self.summary(),
            "linear_counts": self.linear_counts,
            "log10_counts": self.log10_counts,
            "max_linear_counts": self.max_linear_counts,
            "entropy_norm_counts": self.entropy_norm_counts,
            "top1_lag_counts": self.top1_lag_counts,
            "entropy_norm_by_layer_summary": entropy_norm_by_layer_summary,
            "entropy_norm_counts_by_layer": self.entropy_norm_counts_by_layer,
            "top1_lag_by_layer_summary": top1_lag_by_layer_summary,
            "top1_lag_counts_by_layer": self.top1_lag_counts_by_layer,
            "row_std_by_layer_head_summary": row_std_by_layer_head_summary,
        }

    def save(self, save_prefix: str | Path) -> tuple[Path, Path]:
        save_prefix = Path(save_prefix)
        save_prefix.parent.mkdir(parents=True, exist_ok=True)

        pt_path = save_prefix.with_suffix(".pt")
        json_path = save_prefix.with_suffix(".json")

        state = self.state_dict()
        torch.save(state, pt_path)
        payload = dict(state.get("summary") or {})
        payload.update(
            {
                "linear_counts": self.linear_counts.tolist(),
                "log10_counts": self.log10_counts.tolist(),
                "max_linear_counts": self.max_linear_counts.tolist(),
                "entropy_norm_counts": self.entropy_norm_counts.tolist(),
                "top1_lag_counts": self.top1_lag_counts.tolist(),
            }
        )
        if self.config.per_layer:
            payload["entropy_norm_by_layer_summary"] = state.get("entropy_norm_by_layer_summary", {})
            payload["entropy_norm_counts_by_layer"] = {
                str(layer): counts.tolist() for layer, counts in self.entropy_norm_counts_by_layer.items()
            }
            payload["top1_lag_by_layer_summary"] = state.get("top1_lag_by_layer_summary", {})
            payload["top1_lag_counts_by_layer"] = {
                str(layer): counts.tolist() for layer, counts in self.top1_lag_counts_by_layer.items()
            }
        if self.config.per_head:
            payload["row_std_by_layer_head_summary"] = state.get("row_std_by_layer_head_summary", {})
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return pt_path, json_path

    @contextmanager
    def patch(self):
        if (
            self._orig_torch_softmax is not None
            or self._orig_tensor_softmax is not None
        ):
            raise RuntimeError("SoftmaxStatsCollector.patch() called while already patched.")

        self._orig_torch_softmax = torch.softmax
        self._orig_tensor_softmax = torch.Tensor.softmax

        collector = self

        def _wrapped_torch_softmax(input, dim, dtype=None):
            out = collector._orig_torch_softmax(input, dim, dtype=dtype)
            try:
                collector.observe(out, dim=int(dim))
            except Exception:
                collector.n_skipped += 1
            return out

        def _wrapped_tensor_softmax(self, dim, dtype=None):
            out = collector._orig_tensor_softmax(self, dim, dtype=dtype)
            try:
                collector.observe(out, dim=int(dim))
            except Exception:
                collector.n_skipped += 1
            return out

        torch.softmax = _wrapped_torch_softmax
        torch.Tensor.softmax = _wrapped_tensor_softmax
        try:
            yield self
        finally:
            torch.softmax = self._orig_torch_softmax
            torch.Tensor.softmax = self._orig_tensor_softmax
            self._orig_torch_softmax = None
            self._orig_tensor_softmax = None
