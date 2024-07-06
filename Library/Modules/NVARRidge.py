import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

torch.manual_seed(0)

class NVARReservoir(nn.Module):
    def __init__(self, input_size, degree=2, ridge_alpha=0.0):
        super(NVARReservoir, self).__init__()
        self.input_size = input_size
        self.degree = degree
        self.ridge_alpha = ridge_alpha

        # Placeholder for ridge regression weights
        self.Wout = None
        self.Wout_bias = None

    def poly_features(self, X):
        batch_size, seq_len, input_size = X.shape
        features = [X]
        if self.degree > 1:
            for d in range(2, self.degree + 1):
                features.append(torch.pow(X, d))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        # Generate polynomial features
        poly_X = self.poly_features(x)

        if self.Wout is not None:
            outputs = torch.matmul(poly_X, self.Wout.T) + self.Wout_bias
        else:
            outputs = torch.zeros(poly_X.size(0), poly_X.size(1), self.input_size).to(x.device)

        return outputs

    def fit(self, X, y):
        # Generate polynomial features
        poly_X = self.poly_features(X).view(-1, self.poly_features(X).shape[-1]).cpu().numpy()
        y = y.view(-1, self.input_size).cpu().numpy()

        # Perform ridge regression
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(poly_X, y)
        self.Wout = torch.tensor(ridge.coef_, dtype=torch.float32).to(X.device)
        self.Wout_bias = torch.tensor(ridge.intercept_, dtype=torch.float32).to(X.device)