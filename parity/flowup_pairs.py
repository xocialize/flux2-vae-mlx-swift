"""FlowUpscaler parity pairs from the PyTorch reference (TensorForger's upscaler_unet.py, Unlicense), CPU fp32.

For each case: `latents` [1,32,h,w] (unpatchified, bn-normalised: a Klein 4B 1024² draw from the AB-R-0795 eval),
`noise` [1,32,2h,2w] (torch.randn, fixed seed), and the reference's `v` and `out` = noise - v. Cases:
  square    the whole 1024² draw (latents 128², out 256²)
  wide      a 64×96 crop of it (non-square: exercises the bilinear resizes at other ratios)
  odd       a 40×56 crop (the 1/4 stage is 20×28; the U-Net's stride-2 ceil and the up-stage resize to the skip)
  bf16w     square, weights rounded to bf16 first (what the converted file carries), still fp32 compute
usage: python flowup_pairs.py <weights dir> <base latents.safetensors> <out dir>
"""
import sys
import torch
from safetensors.torch import load_file, save_file

wdir, lat_path, out = sys.argv[1:4]
sys.path.insert(0, wdir)
from upscaler_unet import UpscalerUNet  # noqa: E402

torch.set_grad_enabled(False)


def unpatchify(x):
    b, c, h, w = x.shape
    return x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


sd = load_file(f"{wdir}/flow_upscaler.safetensors")
net = UpscalerUNet().eval()
net.load_state_dict(sd)
net_b = UpscalerUNet().eval()
net_b.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in sd.items()})

full = unpatchify(load_file(lat_path)["latents"].float())
cases = {"square": (full, net), "wide": (full[:, :, 16:80, 8:104], net), "odd": (full[:, :, 30:70, 20:76], net),
         "bf16w": (full, net_b)}
for i, (name, (z, model)) in enumerate(cases.items()):
    z = z.contiguous()
    g = torch.Generator().manual_seed(1234 + i)
    noise = torch.randn((1, 32, z.shape[2] * 2, z.shape[3] * 2), generator=g)
    v = model(sample=noise, timestep=torch.full((1,), 1000.0), latents_small=z)
    save_file({"latents": z, "noise": noise, "v": v.contiguous(), "out": (noise - v).contiguous()},
              f"{out}/pair-{name}.safetensors")
    print(name, tuple(z.shape), "->", tuple(v.shape), "out std %.4f" % (noise - v).std().item())
