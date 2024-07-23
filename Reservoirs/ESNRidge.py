import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge, Lasso

class ESNReservoir(nn.Module):
    def __init__(self, io_size, reservoir_size, pred_len, spectral_radius=0.90,
                 ridge_alpha=0.03, leaking_rate=1.0, connectivity=0.1):
        super(ESNReservoir, self).__init__()
        self.io_size = io_size
        self.reservoir_size = reservoir_size
        self.pred_len = pred_len
        self.ridge_alpha = ridge_alpha
        self.leaking_rate = leaking_rate

        # Input weights
        self.Win = nn.Parameter(torch.randn(reservoir_size, io_size) * 0.1, requires_grad=False)

        # Reservoir weights
        W = torch.randn(reservoir_size, reservoir_size)

        # Adjust spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        W *= spectral_radius / max_eigenvalue

        # Apply connectivity
        conn_mask = (torch.rand(reservoir_size, reservoir_size) < connectivity).float()
        W *= conn_mask

        self.W = nn.Parameter(W, requires_grad=False)

        # Placeholder for ridge regression weights
        self.Wout = None
        self.Wout_bias = None

    def forward(self, x, h=None):
        device = x.device
        if h is None:
            h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = x.size(1)

        states = []
        for t in range(input_len):
            input = x[0, t, :].unsqueeze(0)
            h_new = F.tanh(self.Win @ input.T + self.W @ h.T).T
            h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
            states.append(h)

        states = torch.cat(states, dim=0)
        if self.Wout is not None:
            outputs = torch.matmul(states, self.Wout.T) + self.Wout_bias
        else:
            outputs = torch.zeros(states.size(0), self.io_size).to(device)

        return outputs.unsqueeze(0), h

    def fit(self, X, y):
        device = X.device
        h = torch.zeros(1, self.reservoir_size).to(device)
        input_len = X.size(1)
        states = []

        with torch.no_grad():
            for t in range(input_len):
                input = X[0, t, :].unsqueeze(0)
                h_new = F.tanh(self.Win @ input.T + self.W @ h.T).T
                h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
                states.append(h)

        states = torch.cat(states, dim=0).cpu().numpy()
        y = y.cpu().numpy()
        y = y.squeeze(0)

        # Perform ridge regression
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(states, y)
        self.Wout = torch.tensor(ridge.coef_, dtype=torch.float32).to(device)
        self.Wout_bias = torch.tensor(ridge.intercept_, dtype=torch.float32).to(device)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        return torch.matmul(states, self.Wout.T) + self.Wout_bias

    def thermalize(self, X):
        device = X.device

        # generate numbers between 0 and 100
        h = torch.rand(1, self.reservoir_size).to(device)
        input_len = X.size(1)
        states = torch.zeros((1, h.size(1))).to(device)

        with torch.no_grad():
            for t in range(input_len):
                input = X[0, t, :].unsqueeze(0)
                h_new = F.tanh(self.Win @ input.T + self.W @ h.T).T
                h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
                states = torch.cat((states, h), dim=0)

        return states[1:]

