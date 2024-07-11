import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class ESNPCA(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, components=5, spectral_radius=0.9, sparsity=0.5, warmup=100, leaking_rate=0.1):
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

        ## TRAINABLE STUFFS
        ## PCA
        if isinstance(components, int):
            self.ncomp = components
        elif components < 1:
            self.ncomp = int(components*self.reservoir_size)
        else:
            self.ncomp = 5
        self.pca = PCA(self.ncomp)
        self.linreg = LinearRegression()
        self.scaler = StandardScaler()

    def reservoir(self, input, h):
        # input: (1, input_size)
        # h: (1, reservoir_size)
        h_new = F.tanh(self.Win @ input + self.Wh @ h)
        h = self.leaking_rate * h + (1-self.leaking_rate) * h_new
        return h

    def forward(self, x):
        
        device = x.device  
        input_len = x.size(0)

        H = torch.zeros(size=(input_len, self.reservoir_size + self.input_size)).to(device)
        h = torch.zeros(self.reservoir_size).to(device)

        for t in range(input_len):
            # take single point
            input = x[t,:]
            # get hidden state from the point extracted and the previous hidden state
            h = self.reservoir(input, h)
            #if t >= self.warmup:
                #ext_state = torch.cat((h,input))
                #H[t-self.warmup,:] = ext_state  
            ext_state = torch.cat((h,input))
            H[t,:] = ext_state       

        return H

    def fit(self, input, target):
        # get reservoir states
        reservoir_states = self.forward(input)
        device = reservoir_states.device
        # scaler of the H tensor
        reservoir_states = self.scaler.fit_transform(reservoir_states.cpu().numpy())
        # dimensionality reduction
        reservoir_states = torch.tensor(self.pca.fit_transform(reservoir_states)).to(device).float()
        # pseudo inverse of last hidden states (the one of the predictions)
        linreg = self.linreg.fit(reservoir_states.cpu(), target.cpu())
        self.Wout = torch.tensor(linreg.coef_).to(device).float()
        self.bias = torch.tensor(linreg.intercept_).to(device).float()
        # output
        output = reservoir_states @ self.Wout.T + self.bias
        return output

        
    def predict(self, input, reservoir_states=None):
        if reservoir_states is None:
            reservoir_states = self.forward(input)
        device = reservoir_states.device
        # scale the states (it has been fitted before)
        reservoir_states_transformed = self.scaler.transform(reservoir_states.cpu().numpy())
        # dimensionality reduction
        reservoir_states_transformed = self.pca.transform(reservoir_states_transformed)
        reservoir_states_transformed = torch.tensor(reservoir_states_transformed).to(device).float()
        # get last hidden state to predict the new point -> size = (1, ncomp))
        h = reservoir_states_transformed[-1]
        # calculate outputs
        output = h @ self.Wout.T + self.bias
        output = output.squeeze(0)
        # calculate the hidden state of this new prediction using the original hidden state
        h = self.reservoir(output, reservoir_states[-1, :-self.input_size])

        return output, h
    
    def generate(self, input, pred_len):
        device = input.device
        outputs = torch.zeros(size=(pred_len, self.output_size)).to(device)
        ext_state, h = None, None
        x = input
        for t in range(pred_len):
            if h is not None:
                ext_state = torch.cat((h,x)).unsqueeze(0)
            x, h = self.predict(x, ext_state)
            outputs[t,:]=x
        return outputs
