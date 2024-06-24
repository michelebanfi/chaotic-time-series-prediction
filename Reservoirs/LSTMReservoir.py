import torch.nn as nn
import torch.nn.functional as F

class LSTMReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(LSTMReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # LSTM as reservoir
        self.lstm = nn.LSTM(input_size, reservoir_size, num_layers=2, batch_first=True)

        # Output weights
        self.linear1 = nn.Linear(reservoir_size, 64)
        self.linear2 = nn.Linear(64, output_size)


        # Freeze LSTM parameters
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.25)

    # LSTM forward pass
    def forward(self, x):
        h, _ = self.lstm(x)

        y = F.leaky_relu(self.linear1(h))
        y = self.dropout(y)
        y = self.linear2(y)
        return y