import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, num_layers=1, pred_len=1):
        super(LSTMReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        # LSTM as reservoir
        self.lstm = nn.LSTM(input_size, reservoir_size, num_layers=num_layers, batch_first=True)
        # freeze reservoir
        for param in self.lstm.parameters():
            param.requires_grad = False

        # Output weights
        self.linear1 = nn.Linear(reservoir_size, 512)
        self.linear2 = nn.Linear(512, output_size)

        self.dropout = nn.Dropout(0.2)

    # LSTM forward pass
    def forward(self, x):

        # input shape: (amount of sequences, sequences length, dimensionality of problem)
        input_len = x.size(1)
        for i in range(self.pred_len):
            # get the input and the previous outputs
            input = x[:, i:i+input_len, :]

            # the output will be just on the last hidden state
            h, _ = self.lstm(input)
            h = h[:, -1, :]
            out = F.leaky_relu(self.linear1(h))
            out = self.dropout(out)
            out = self.linear2(out)

            out = out.unsqueeze(1)
            x = torch.cat((x, out), dim=1)

        x = x[:, -self.pred_len:, :]
        return x