import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, seq_len=1):
        super(GRUReservoir, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.seq_len = seq_len

        # LSTM as reservoir
        self.gru = nn.GRU(input_size, reservoir_size, num_layers=2, batch_first=True)

        # Output weights
        self.linear1 = nn.Linear(reservoir_size, 64)
        self.linear2 = nn.Linear(64, output_size)


        # Freeze LSTM parameters
        for param in self.gru.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.2)

    # LSTM forward pass
    def forward(self, x):

        z = torch.zeros(x.size(0) + self.seq_len, 1, self.output_size, dtype=torch.float32)
        z[:x.size(0), :, :] = x[:, :, :]
        for i in range(self.seq_len):
            input = z[i:i + x.size(0), :, :]
            h, _ = self.gru(input)

            h = h[-1, :, :]

            out = F.leaky_relu(self.linear1(h))
            out = self.dropout(out)
            out = self.linear2(out)

            out = out.unsqueeze(1)
            x = torch.cat((x, out), 0)

        x = x[-self.seq_len:, :, :]
        x = x.transpose(0, 1)
        return x