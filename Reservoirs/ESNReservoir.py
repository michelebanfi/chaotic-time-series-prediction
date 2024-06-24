import torch
import torch.nn as nn
import torch.nn.functional as F

class ESNReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, sparsity=0.1):
        super(ESNReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

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
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.reservoir_size)
        outputs = []

        for t in range(seq_len):
            h = torch.tanh(self.Win @ x[:, t, :].T + self.W @ h.T).T
            outputs.append(self.Wout(h))

        outputs = torch.stack(outputs, dim=1)
        return outputs