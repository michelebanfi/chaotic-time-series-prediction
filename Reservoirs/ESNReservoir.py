import torch
import torch.nn as nn
import torch.nn.functional as F

class ESNReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, seq_len, spectral_radius=0.9, sparsity=0.1):
        super(ESNReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.seq_len = seq_len

        # Input weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, input_size))

        # Reservoir weights
        self.W = nn.Parameter(torch.randn(reservoir_size, reservoir_size))

        # Output weights
        self.Wout = nn.Linear(reservoir_size, output_size)

        # Adjust spectral radius
        self.W.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W.data)))

        # Apply sparsity
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        self.W.data *= mask

    def forward(self, x):
        h = torch.zeros(1, self.reservoir_size)

        for t in range(self.seq_len):
            h = torch.tanh(self.Win @ x[:, t, :].T + self.W @ h.T).T
            output = self.Wout(h)
            output = output.unsqueeze(1)
            x = torch.cat((x, output), dim=1)

        return x[:, -self.seq_len:, :]