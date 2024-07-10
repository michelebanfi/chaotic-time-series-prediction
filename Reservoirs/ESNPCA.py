import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ESNPCA(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, components=5,spectral_radius=0.9, sparsity=0.1, warmup=100, leaking_rate=0.01):
        super(ESNPCA, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.warmup = warmup
        self.leaking_rate = leaking_rate

        ## NON TRAINABLE PARAMETERS
        # Input weights
        self.Win = nn.Parameter(torch.rand(reservoir_size, input_size)-0.5, requires_grad=False)

        # Reservoir weights
        self.Wh = nn.Parameter(torch.rand(reservoir_size, reservoir_size)-0.5, requires_grad=False)
        # Adjust spectral radius
        self.Wh.data *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.Wh.data)))
        # Apply sparsity
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        self.Wh.data *= mask

        ## TRAINABLE OUTPUT
        ## PCA
        if isinstance(components, int):
            self.ncomp = components
        elif components < 1:
            self.ncomp = int(components*self.reservoir_size)
        else:
            self.ncomp = 5
        self.pca = PCA(self.ncomp)
        # Output weights
        self.Wout = nn.Linear(self.ncomp, output_size)



    def forward(self, x):
        
        device = x.device  
        input_len = x.size(0)

        H = torch.zeros(size=(input_len - self.warmup, self.reservoir_size + self.input_size)).to(device)
        h = torch.zeros(self.reservoir_size).to(device)

        for t in range(input_len):
            # take single point
            input = x[t,:]
            # get hidden state from the point extracted and the previous hidden state
            h_new = F.tanh(self.Win @ input + self.Wh @ h)
            h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
            if t >= self.warmup:
                ext_state = torch.cat((h,input))
                H[t-self.warmup,:] = ext_state
                    

        return H

    def fit(self, input):
        # get reservoir states
        reservoir_states = self.forward(input[:-1])
        device = reservoir_states.device
        # scaler
        scaler = StandardScaler()
        reservoir_states = scaler.fit_transform(reservoir_states.cpu().numpy())
        # dimensionality reduction
        reduced_states = torch.tensor(self.pca.fit_transform(reservoir_states)).to(device).float()
        print(f"Components: {self.ncomp}", f"- Explained variance: {100*self.pca.explained_variance_ratio_.sum():.1f} %")
        # pseudo inverse of last hidden states (the one of the predictions)
        pinv = torch.pinverse(reduced_states)
        # the input is the target if shifted by 1 (and the warmup)
        target = input[self.warmup+1:]
        self.Wout.data = (pinv @ target).T

        
    def predict(self, input):
        reservoir_states = self.forward(input[:-1])
        device = reservoir_states.device
        # scaler
        scaler = StandardScaler()
        reservoir_states = scaler.fit_transform(reservoir_states.cpu().numpy())
        # dimensionality reduction
        reduced_states = torch.tensor(self.pca.transform(reservoir_states)).to(device).float()
        # calculate outputs
        output = reduced_states @ self.Wout.data.T

        return output