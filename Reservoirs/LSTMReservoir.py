import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, seq_len=1):
        super(LSTMReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.seq_len = seq_len

        # LSTM as reservoir
        self.lstm = nn.LSTM(input_size, reservoir_size, num_layers=2, batch_first=True)

        # Output weights
        self.linear1 = nn.Linear(reservoir_size, 64)
        self.linear2 = nn.Linear(64, output_size)


        # Freeze LSTM parameters
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0)

    # LSTM forward pass
    def forward(self, x):

        for i in range(self.seq_len):
            z = x[:, i, :].unsqueeze(1)
            h, _ = self.lstm(z)

            out = F.leaky_relu(self.linear1(h))
            out = self.dropout(out)
            out = self.linear2(out)
            x = torch.cat((x, out), 1)

        return x[:, self.seq_len - 1:, :]