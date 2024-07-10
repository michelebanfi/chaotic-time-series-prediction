import torch
import torch.nn as nn
import torch.nn.functional as F

class ESNReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, pred_len, spectral_radius=0.8, sparsity=0.1):
        super(ESNReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.pred_len = pred_len

        # Input weights
        self.Win = nn.Parameter(torch.rand(reservoir_size, input_size)-0.5, requires_grad=False)

        # Reservoir weights
        self.W = nn.Parameter(torch.rand(reservoir_size, reservoir_size)-0.5, requires_grad=False)

        # Output weights
        self.Wout = nn.Linear(reservoir_size, output_size)

        # Adjust spectral radius
        # multiply W by spectral radius divided by max of eigenvalues
        self.W.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W.data)))

        # Apply sparsity
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        self.W.data *= mask


    def forward(self, x):
        device = x.device
        h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = x.size(1)

        for t in range(input_len + self.pred_len-1):
            # take single point
            input = x[0, t, :]
            input = input.unsqueeze(0)
            # get hidden state from the point extracted and the previous hidden state
            h = F.leaky_relu(self.Win @ input.T + self.W @ h.T).T
            # if the time reached the input lenght we start concatenating outputs in order to pick them in the next round
            if t >= input_len-1:
                # calculate the output
                output = self.Wout(h)
                output = output.unsqueeze(1)
                x = torch.cat((x, output), dim=1)

        return x[:, -self.pred_len:, :]