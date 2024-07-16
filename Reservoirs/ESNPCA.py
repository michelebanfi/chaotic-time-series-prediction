import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class ESNPCA(nn.Module):
    def __init__(self, input_size, output_size, reservoir_size=512, components=0.05, spectral_radius=0.9, sparsity=0.5, warmup=100, leaking_rate=0.1, seed=None):
        super(ESNPCA, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.warmup = warmup
        self.leaking_rate = leaking_rate

        if seed is not None:
            torch.manual_seed(seed)

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
        if components > 0 and isinstance(components, int):
            self.ncomp = components
        elif components > 0 and components < 1:
            self.ncomp = int(components*self.reservoir_size)
            if self.ncomp == 0:
                self.ncomp += 1
        else:
            self.ncomp = 0.1
        self.pca = PCA(self.ncomp)
        self.linreg = LinearRegression()
        self.scaler = StandardScaler()

    def reservoir(self, input, h):
        # input: (input_size,dimensionality)
        # h: (reservoir_size,1)
        h_new = F.tanh(self.Win @ input + self.Wh @ h)
        h = (1-self.leaking_rate) * h + self.leaking_rate * h_new
        return h
    
    def output(self, reservoir_states):        
        return reservoir_states @ self.Wout.T + self.bias
    
    def thermalize(self, input):
        device = input.device
        # input of shape=(warmup,dimensionality)
        H = torch.zeros(size=(self.warmup+1, self.reservoir_size), device=device)
        for t in range(self.warmup):
            h = H[t]
            x = input[t]
            H[t+1] = self.reservoir(x, h)
        return H[1:]

    def forward(self, x, h):
        
        # device and input lenght
        device = x.device  
        input_len = x.size(0)

        # store extended states
        H = torch.zeros(size=(input_len, self.reservoir_size + self.input_size)).to(device)

        for t in range(input_len):
            # take single point
            input = x[t]
            # get hidden state from the point extracted and the previous hidden state
            h = self.reservoir(input, h)
            # storing extended states
            ext_state = torch.cat((h,input))
            H[t] = ext_state  

        return H

    def fit(self, input, target):
        device = input.device
        # organize data for warmup
        target = target[self.warmup:]
        warmup_input = input[:self.warmup]
        input = input[self.warmup:]
        # warmup process: take last hidden state
        h = self.thermalize(warmup_input)[-1]
        # get reservoir states: (n_input - warmup, res_size + dimensionality)
        reservoir_states = self.forward(input, h)
        # fit the scaler on the H tensor
        reservoir_states = self.scaler.fit_transform(reservoir_states.cpu().numpy())
        # dimensionality reduction
        reservoir_states = self.pca.fit_transform(reservoir_states) # output size: (n_input - warmup, ncomponents)
        reservoir_states = torch.tensor(reservoir_states).to(device).float()
        # simple linear regression on PCA components
        # reservoir_states are used as (n_input - warmup) samples of dimension (ncomponents) 
        linreg = self.linreg.fit(reservoir_states.cpu(), target.cpu())
        self.Wout = torch.tensor(linreg.coef_).to(device).float()
        self.bias = torch.tensor(linreg.intercept_).to(device).float()
        # output
        output = self.output(reservoir_states)
        return output, target

        
    def predict(self, input, extended_states=None):
        
        device = input.device
        if extended_states is None:
            # organize data for warmup
            warmup_input = input[:self.warmup]
            input = input[self.warmup:]
            # warmup process: take last hidden state
            h = self.thermalize(warmup_input)[-1]
            # if no starting hidden state is provided calculate them
            extended_states = self.forward(input, h)

        ## Generate new prediction
        # scale the states for PCA
        extended_states_processed = self.scaler.transform(extended_states.cpu().numpy())
        # take the last for prediction
        extended_states_processed = np.expand_dims(extended_states_processed[-1], axis=0)
        # dimensionality reduction
        extended_states_processed = self.pca.transform(extended_states_processed)
        extended_states_processed = torch.tensor(extended_states_processed).to(device).float()
        # calculate outputs
        output = self.output(extended_states_processed)

        ## Create the new vector of extended states
        # calculate the hidden state of this new prediction using the original hidden state 
        output = output.squeeze(0)
        # get ORIGINAL hidden state that produced output
        hidden_state = extended_states[-1, :-self.input_size]
        # calculate the hidden state generated by output
        hidden_state = self.reservoir(output, hidden_state)
        # create the new extended state
        ext_state = torch.cat((hidden_state, output)).unsqueeze(0)
        # concat the new extended state (RETURN IT ONLY IF USE fit_transform IN SCALER)
        # extended_states = torch.cat((extended_states, ext_state))

        return output, ext_state
    
    def generate(self, input, pred_len):
        device = input.device
        outputs = torch.zeros(size=(pred_len, self.output_size)).to(device)
        ext_states = None
        x = input
        for t in range(pred_len):
            x, ext_states = self.predict(x, ext_states)
            outputs[t,:]=x
        return outputs
