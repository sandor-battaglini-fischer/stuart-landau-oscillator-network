import math

import torch


class HORN(torch.nn.Module):

    def __init__(
        self,
        num_input,
        num_nodes,
        num_output,
        h,
        alpha,
        omega,
        gamma,
        use_tanh=True,
    ):
        super().__init__()

        self.num_input = num_input
        self.num_nodes = num_nodes
        self.num_output = num_output
        self.h = h
        self.alpha = alpha
        self.omega = omega
        self.gamma = gamma
        self.use_tanh = use_tanh

        self.omega_factor = omega * omega
        self.gamma_factor = 2.0 * gamma
        self.gain_rec = 1.0 / math.sqrt(self.num_nodes)

        self.i2h = torch.nn.Linear(num_input, num_nodes)
        self.h2h = torch.nn.Linear(num_nodes, num_nodes)
        self.h2o = torch.nn.Linear(num_nodes, num_output)

    def dynamics_step(self, x_t, y_t, input_t):
        pre_act = self.i2h(input_t) + self.gain_rec * self.h2h(y_t)
        if self.use_tanh:
            forcing = self.alpha * torch.tanh(pre_act)
        else:
            forcing = self.alpha * pre_act

        y_t = y_t + self.h * (
            forcing
            - self.omega_factor * x_t
            - self.gamma_factor * y_t
        )
        x_t = x_t + self.h * y_t
        return x_t, y_t

    def forward(self, batch, random_init=None, record=False):
        batch_size = batch.size(1)
        num_timesteps = batch.size(0)

        ret = {}

        if record:
            rec_x_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            rec_y_t = torch.zeros(batch_size, num_timesteps, self.num_nodes)
            ret["rec_x_t"] = rec_x_t
            ret["rec_y_t"] = rec_y_t

        if random_init is not None:
            x_t = torch.randn(batch_size, self.num_nodes) * random_init
            y_t = torch.randn(batch_size, self.num_nodes) * random_init
        else:
            x_t = torch.zeros(batch_size, self.num_nodes)
            y_t = torch.zeros(batch_size, self.num_nodes)

        for t in range(num_timesteps):
            x_t, y_t = self.dynamics_step(x_t, y_t, batch[t])

            if record:
                rec_x_t[:, t, :] = x_t
                rec_y_t[:, t, :] = y_t

        output = self.h2o(x_t)
        ret["output"] = output
        return ret
