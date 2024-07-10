import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

torch.manual_seed(0)


class NVARReservoir(nn.Module):
    def __init__(self, io_size, degree=2, ridge_alpha=0.0, delay=1, stride=1):
        super(NVARReservoir, self).__init__()
        self.input_size = io_size
        self.degree = degree
        self.ridge_alpha = ridge_alpha
        self.delay = delay
        self.stride = stride
        # Placeholder for ridge regression weights
        self.Wout = None
        self.Wout_bias = None

    def poly_features(self, X):
        if X.dim() == 2:
            batch_size, input_size = X.shape
            X = X.unsqueeze(1)  # Add sequence dimension
        elif X.dim() == 3:
            batch_size, seq_len, input_size = X.shape
        else:
            raise ValueError("Input must be 2D or 3D tensor")

        features = [X]
        if self.degree > 1:
            for d in range(2, self.degree + 1):
                features.append(torch.pow(X, d))
        return torch.cat(features, dim=-1)

    def create_delay_features(self, X):
        if X.dim() == 2:
            X = X.unsqueeze(0)  # Add batch dimension
        batch_size, seq_len, feature_size = X.shape
        delayed_features = []

        for d in range(self.delay):
            if d == 0:
                delayed_features.append(X)
            else:
                pad = torch.zeros(batch_size, d, feature_size).to(X.device)
                delayed = torch.cat([pad, X[:, :-d, :]], dim=1)
                delayed_features.append(delayed)

        return torch.cat(delayed_features, dim=-1)

    def forward(self, x):
        # Generate polynomial features
        poly_X = self.poly_features(x)

        # Create delayed features
        delayed_X = self.create_delay_features(poly_X)

        # Apply stride
        strided_X = delayed_X[:, ::self.stride, :]

        if self.Wout is not None:
            outputs = torch.matmul(strided_X, self.Wout.T) + self.Wout_bias
        else:
            outputs = torch.zeros(strided_X.size(0), strided_X.size(1), self.input_size).to(x.device)
        return outputs

    def fit(self, X, y):
        # Generate polynomial features
        poly_X = self.poly_features(X)

        # Create delayed features
        delayed_X = self.create_delay_features(poly_X)

        # Apply stride to both X and y
        strided_X = delayed_X[:, ::self.stride, :]
        strided_y = y[:, ::self.stride, :]

        # Adjust the length of strided_y to match strided_X
        min_len = min(strided_X.size(1), strided_y.size(1))
        strided_X = strided_X[:, :min_len, :]
        strided_y = strided_y[:, :min_len, :]

        # Reshape for ridge regression
        X_fit = strided_X.reshape(-1, strided_X.shape[-1]).cpu().numpy()
        y_fit = strided_y.reshape(-1, self.input_size).cpu().numpy()

        # Perform ridge regression
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(X_fit, y_fit)
        self.Wout = torch.tensor(ridge.coef_, dtype=torch.float32).to(X.device)
        self.Wout_bias = torch.tensor(ridge.intercept_, dtype=torch.float32).to(X.device)