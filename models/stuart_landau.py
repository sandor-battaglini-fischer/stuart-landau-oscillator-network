# SL-RNN (Stuart-Landau Recurreent Neural Network) module

import math
import torch

class SLON(torch.nn.Module):

    def __init__(
        self,
        num_input,
        num_nodes,
        num_output,
        h,
        alpha,
        omega,
        gamma,
        lambda_param=None,
        gamma_real=None,
        gamma_imag=None,
        use_tanh=True,
        linear_dynamics=False,
    ):
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output

        self.h = h
        self.alpha = alpha
        self.use_tanh = use_tanh
        self.linear_dynamics = linear_dynamics

        if lambda_param is None:
            self.lambda_param = torch.ones(num_nodes) * (-abs(gamma))
        else:
            self.lambda_param = lambda_param if isinstance(lambda_param, torch.Tensor) else torch.ones(num_nodes) * lambda_param
        
        if isinstance(omega, torch.Tensor):
            self.omega_param = omega
        else:
            self.omega_param = torch.ones(num_nodes) * omega
        
        if gamma_real is None:
            self.gamma_real = torch.ones(num_nodes) * (-0.1)
        else:
            self.gamma_real = gamma_real if isinstance(gamma_real, torch.Tensor) else torch.ones(num_nodes) * gamma_real
        
        if gamma_imag is None:
            self.gamma_imag = torch.ones(num_nodes) * 0.0
        else:
            self.gamma_imag = gamma_imag if isinstance(gamma_imag, torch.Tensor) else torch.ones(num_nodes) * gamma_imag

        self.gain_rec = 1. / math.sqrt(self.num_nodes)

        self.i2h = torch.nn.Linear(num_input, num_nodes)
        self.h2h = torch.nn.Linear(num_nodes * 2, num_nodes)
        self.h2o = torch.nn.Linear(num_nodes * 2, num_output)

    def dynamics_step(self, z_real, z_imag, input_t):
        z = torch.complex(z_real, z_imag)
        
        lambda_omega_coeff = torch.complex(self.lambda_param, self.omega_param)
        gamma_coeff = torch.complex(self.gamma_real, self.gamma_imag)
        
        z_state = torch.cat([z_real, z_imag], dim=1)
        
        pre_act = self.i2h(input_t) + self.gain_rec * self.h2h(z_state)
        if self.use_tanh:
            input_force = self.alpha * torch.tanh(pre_act)
        else:
            input_force = self.alpha * pre_act

        if self.linear_dynamics:
            dz_dt = lambda_omega_coeff * z + torch.complex(input_force, torch.zeros_like(input_force))
        else:
            dz_dt = (
                lambda_omega_coeff * z
                + gamma_coeff * torch.abs(z) ** 2 * z
                + torch.complex(input_force, torch.zeros_like(input_force))
            )
        
        z = z + self.h * dz_dt
        
        return torch.real(z), torch.imag(z)

    def forward(self, batch, random_init = None, record = False):
        batch_size = batch.size(1)
        num_timesteps = batch.size(0)

        ret = {}

        if record:
            rec_z_real = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_z_imag = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            ret['rec_z_real'] = rec_z_real
            ret['rec_z_imag'] = rec_z_imag

        if not random_init is None:
            z_real_0 = torch.randn(batch_size, self.num_nodes) * random_init
            z_imag_0 = torch.randn(batch_size, self.num_nodes) * random_init
        else:
            z_real_0 = torch.zeros(batch_size, self.num_nodes)
            z_imag_0 = torch.zeros(batch_size, self.num_nodes)

        z_real = torch.autograd.Variable(z_real_0)
        z_imag = torch.autograd.Variable(z_imag_0)

        for t in range(num_timesteps):
            z_real, z_imag = self.dynamics_step(z_real, z_imag, batch[t])

            if record:
                rec_z_real[:, t, :] = z_real
                rec_z_imag[:, t, :] = z_imag

        z_features = torch.cat([z_real, z_imag], dim=1)
        output = self.h2o(z_features)

        ret['output'] = output
        return ret
