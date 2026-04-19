# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 CARIAD SE
import random
import numpy as np
import torch
import torchvision.transforms.functional as F

def _np_rgb_to_t(rgb_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_np.astype(np.uint8)).permute(2, 0, 1)

def _t_rgb_to_np(rgb_t: torch.Tensor) -> np.ndarray:
    if rgb_t.dtype != torch.uint8:
        rgb_t = rgb_t.clamp(0, 255).byte()
    return rgb_t.permute(1, 2, 0).cpu().numpy()

class TorchColorAug:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.8):
        self.b,self.c,self.s,self.h,self.p = brightness,contrast,saturation,hue,p
    def __call__(self, rgb_np: np.ndarray, *rest):
        if random.random() > self.p:
            return (rgb_np, *rest)
        x = _np_rgb_to_t(rgb_np).float() / 255.0
        x = F.adjust_brightness(x, 1.0 + random.uniform(-self.b, self.b))
        x = F.adjust_contrast(x,  1.0 + random.uniform(-self.c, self.c))
        x = F.adjust_saturation(x, 1.0 + random.uniform(-self.s, self.s))
        x = F.adjust_hue(x,        random.uniform(-self.h, self.h))
        x = (x * 255.0).clamp(0, 255)
        return (_t_rgb_to_np(x), *rest)

class TorchGeoAug:
    def __init__(self, hflip_p=0.5):
        self.hflip_p = hflip_p
    def __call__(self, rgb_np: np.ndarray, *depths_and_K):
        *depths, K = depths_and_K
        H, W = rgb_np.shape[:2]
        do = random.random() < self.hflip_p
        if not do:
            return (rgb_np, *depths, K)
        # flip rgb
        rgb_t = _np_rgb_to_t(rgb_np)
        rgb_t = F.hflip(rgb_t)
        rgb_np2 = _t_rgb_to_np(rgb_t)
        # flip depths
        out_depths = []
        for d in depths:
            if d is None:
                out_depths.append(None); continue
            t = torch.from_numpy(d.transpose(2,0,1))
            t = F.hflip(t)
            out_depths.append(t.permute(1,2,0).numpy())
        # adjust K for horizontal flip
        K2 = None if K is None else K.copy()
        if K2 is not None:
            K2[0,2] = (W - 1) - K2[0,2]
        return (rgb_np2, *out_depths, K2)

class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms or []
    def __call__(self, *args):
        out = args
        for t in self.transforms:
            out = t(*out)
        return out
