import os
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models import SLON

torch.manual_seed(1)
torch.set_grad_enabled(False)

num_input = 1
num_hidden = 32
num_output = 10

h = 1.0
alpha = 0.04
omega_base = 2.0 * torch.pi / 28.0
gamma_base = 0.01

omega_min = 0.5 * omega_base
omega_max = 2.0 * omega_base
omega = torch.rand(num_hidden) * (omega_max - omega_min) + omega_min

model = SLON(
    num_input, num_hidden, num_output, h, alpha, omega, gamma_base,
    lambda_param=-0.05, gamma_real=-0.1, gamma_imag=0.0,
)
model.eval()

model.i2h.weight[:, :] = torch.randn(num_hidden, 1)
model.i2h.bias[:] = 0
model.h2h.weight[:, :] = torch.randn(num_hidden, num_hidden) * 0.1
model.h2h.bias[:] = 0

plt.matshow(model.h2h.weight.detach().numpy())
plt.colorbar()
plt.title("W_hh")

domain = torch.arange(1000)
stimulus = torch.sin(domain * torch.pi * 2 / 60).unsqueeze(0).unsqueeze(0)
stimulus[0, 0, 500:] = 0
stimulus = stimulus.permute(2, 0, 1)

random_init = 1.0
out = model.forward(stimulus, random_init=random_init, record=True)
z_real = out["rec_z_real"]
z_imag = out["rec_z_imag"]
amplitude = torch.sqrt(z_real**2 + z_imag**2)

plt.figure()
for i in range(num_hidden):
    plt.plot(domain, amplitude[0, :, i])
plt.plot(domain, stimulus[:, 0, 0], color="k", linewidth=2)
plt.xlabel("time")
plt.ylabel("amplitude")
plt.show()
