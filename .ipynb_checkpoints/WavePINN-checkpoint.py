import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
import io
import imageio
from torch.nn.parallel import DataParallel
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
device = "cuda"
n_gpus = torch.cuda.device_count()
print(f"GPUs disponibles : {n_gpus}")


def source(x, y, t, cx=0, cy=0, sigma=1., amplitude_g=1e2, f=1.0, amplitude_r=1.0, t0=1):
    gaussian = amplitude_g * torch.exp(-((x - cx)**2/(2*sigma**2)+(y - cy)**2/(2*sigma**2)))
    pi2f2tau2 = (torch.pi * f)**2 * (t - t0)**2
    ricker = amplitude_r * (1.0 - 2.0 * pi2f2tau2) * torch.exp(-pi2f2tau2)
    return gaussian * ricker

c_map = np.load("Density_Test.npy")[8].reshape((51, 51))
c_map = zoom(c_map, 1024/51)

coords = np.linspace(-10, 10, 1024)
interp = RegularGridInterpolator((coords, coords), c_map, method='nearest', bounds_error=False, fill_value=None)

def c(x, y, t=0):
    if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
    return torch.tensor(interp(np.stack([y.ravel(), x.ravel()], axis=-1)).reshape(x.shape), dtype=torch.float32).to(device)

def c_constant(x, y, t=0):
    if type(y) == np.ndarray:
        y = torch.tensor(y)
    if type(x) == np.ndarray:
        x = torch.tensor(x)
    return 10*torch.ones_like(y)

def g(t, t0, alpha=5.0):
    return 0.5 * (torch.tanh(alpha * (t + t0)) + 1)

class PINN(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(6, width), nn.Mish())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.Tanh())
            for _ in range(depth - 1)
        ])
        self.output_layer = nn.Linear(width, 1)

    def forward(self, x, y, t, x0, y0, t0):
        inp = torch.cat([x, y, t, x0, y0, t0], dim=-1)
        h = self.input_layer(inp)
        for layer in self.hidden_layers:
            h = layer(h) + h
        p = self.output_layer(h)
        return p * g(t - t0, t0 / 2)

def loss_fn(model, x, y, t, x0, y0, t0):
    pred = model(x, y, t, x0, y0, t0)
    s_xyt = source(x, y, t, cx=x0, cy=y0, t0=t0)
    ones = torch.ones_like(pred)
    dp_dx, dp_dy, dp_dt = torch.autograd.grad(pred, [x, y, t], grad_outputs=ones, create_graph=True)
    dp_dxx = torch.autograd.grad(dp_dx, x, grad_outputs=ones, create_graph=True)[0]
    dp_dyy = torch.autograd.grad(dp_dy, y, grad_outputs=ones, create_graph=True)[0]
    dp_dtt = torch.autograd.grad(dp_dt, t, grad_outputs=ones, create_graph=True)[0]
    pde_res = dp_dtt - (c(x, y, t) ** 2) * (dp_dxx + dp_dyy) - s_xyt
    return (pde_res ** 2).mean()

model = PINN(32, 4).to(device)

if n_gpus > 1:
    model = DataParallel(model)

V             = 1.0
N_POINTS      = int(1e5)
SPATIAL_RANGE_X  = 50.0
SPATIAL_RANGE_Y  = 50.0
SPATIAL_RANGE_X0 = 10.0
SPATIAL_RANGE_Y0 = 10.0
T0_MAX        = 3.0
T_MAX         = 8.0
N_STEPS       = 300_000
LR            = 1e-3
CKPT_EVERY    = 5000
WAVE_SPEED    = 10.0
FRAC_CAUSAL   = 0.9
T_WINDOW      = T_MAX - T0_MAX

n_causal  = int(N_POINTS * FRAC_CAUSAL)
n_acausal = N_POINTS - n_causal

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=N_STEPS,
    pct_start=0.1, div_factor=1e1, final_div_factor=1e4
)

def sample_batch():
    x  = (torch.rand(N_POINTS, 1, device=device) - 0.5) * 2 * SPATIAL_RANGE_X
    y  = (torch.rand(N_POINTS, 1, device=device) - 0.5) * 2 * SPATIAL_RANGE_Y
    x0 = (torch.rand(N_POINTS, 1, device=device) - 0.5) * 2 * SPATIAL_RANGE_X0
    y0 = (torch.rand(N_POINTS, 1, device=device) - 0.5) * 2 * SPATIAL_RANGE_Y0
    t0 = torch.rand(N_POINTS, 1, device=device) * T0_MAX + 1.0

    dist     = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    t_arrive = t0 + dist / WAVE_SPEED

    t_causal  = t_arrive[:n_causal] + torch.rand(n_causal, 1, device=device) * T_WINDOW
    t_causal  = torch.clamp(t_causal, 0.0, T_MAX)
    t_acausal = torch.rand(n_acausal, 1, device=device) * T_MAX
    t = torch.cat([t_causal, t_acausal])
    return x, y, t, x0, y0, t0

losses = []

for step in (pbar := tqdm(range(N_STEPS))):
    x, y, t, x0, y0, t0 = sample_batch()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss_pde = loss_fn(model, x, y, t, x0, y0, t0)
    loss_pde.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    losses.append(loss_pde.item())
    if step % 20 == 0:
        pbar.set_description(f"loss={np.mean(losses[-20:]):.3e}  lr={scheduler.get_last_lr()[0]:.3e}")

torch.save(model, "model.pt")