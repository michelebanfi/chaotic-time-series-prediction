import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

# create a simple class for the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # LSTM as reservoir
        self.lstm = nn.LSTM(input_size, reservoir_size, num_layers=2, batch_first=True)

        # freeze LSTM parameters
        for param in self.lstm.parameters():
            param.requires_grad = False

        # Output weights
        self.linear1 = nn.Linear(reservoir_size, 64)
        self.linear2 = nn.Linear(64, output_size)

        self.dropout = nn.Dropout(0.2)

    # LSTM forward pass
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[-1:, :]
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# load the model
model = LSTM(3, 128, 3)

# load the data
df = pd.read_csv('../Lorenz/Data/lorenz_data.csv')
data = df[['x', 'y', 'z']].values
t = df['time'].values

# split in train and test
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]
train_t = t[:int(0.8*len(t))]
test_t = t[int(0.8*len(t)):]

# define the sequence length
seq_len = 1

# train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    for i in range(len(train_data)-seq_len):
        x = torch.tensor(train_data[i]).float().unsqueeze(0)
        y = torch.tensor(train_data[i+seq_len]).float().unsqueeze(0)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # test the model
    with torch.no_grad():
        for i in range(len(test_data)-seq_len):
            x = torch.tensor(test_data[i]).float().unsqueeze(0)
            y = torch.tensor(test_data[i+seq_len]).float().unsqueeze(0)

            output = model(x)
            loss = criterion(output, y)

        print(f'Test Loss: {loss.item()}')


# create a plot the predictions of each variable
with torch.no_grad():
    predictions = []
    for i in range(len(test_data)-seq_len):
        x = torch.tensor(test_data[i]).float().unsqueeze(0)
        output = model(x)
        predictions.append(output.numpy()[0])

    predictions = np.array(predictions)
    plt.plot(test_t[1:], predictions[:, 0], label='Predicted x')
    plt.plot(test_t[1:], predictions[:, 1], label='Predicted y')
    plt.plot(test_t[1:], predictions[:, 2], label='Predicted z')
    plt.plot(test_t, test_data[:, 0], label='True x')
    plt.plot(test_t, test_data[:, 1], label='True y')
    plt.plot(test_t, test_data[:, 2], label='True z')
    plt.legend()
    plt.show()

