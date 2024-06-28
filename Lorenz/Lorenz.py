import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time

import sys
sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Utils")
from DataEvaluator import evaluate
from DataLoader import loadData

sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Benchmarks")
from GRU import GRU

sys.path.append("D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Reservoirs")
from GRUReservoir import GRUReservoir

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working on:", device)
print(30*"-")

# Load data from CSV
df = pd.read_csv('D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Data/lorenz_data.csv')
data = df[['x', 'y', 'z']].values
data = torch.tensor(df[['x', 'y','z']].values)

t = df['time'].values

# Define sequence length
pred_len = 100
input_len = 400

# Define the model parameters
io_size = 3
reservoir_size = 128
batch_size = 10
num_epochs = 3 # it will break with 0 epochs

# Load the data
print("Loading data...")
train_t, train_dataloader, val_t, val_dataloader = loadData(data, t, pred_len, input_len)
print("Train batches:", len(train_dataloader))
print("Train input sequences:", len(train_dataloader.dataset))
print("Validation batches:", len(val_dataloader))
print("Validation input sequences:", len(val_dataloader.dataset))
print(30*"-")


# init the models
model = GRUReservoir(io_size, reservoir_size, io_size, pred_len=pred_len, num_layers=2).to(device)
modelBenchmark = GRU(io_size, reservoir_size, io_size, pred_len=pred_len, num_layers=2).to(device)


# NMSE weighted as criterion
def NormalizedMeanSquaredError(y_pred, y_true):
    device = y_pred.get_device()
    pred_len = y_pred.size(1)
    batch_size = y_pred.size(0)

    squared_dist = torch.sum((y_true - y_pred)** 2, dim=2) # squared euclidean distances between predictions
    true_squared_norm = torch.sum(y_true ** 2, dim=2)
    nmse = squared_dist / true_squared_norm
    # actual (from above) shape: (batch size, prediction length)
    # as a neutral transformation for an overall error just take the mean on the prediction length and then on the batch size
    # WEIGHTED
    weights = torch.arange(start=1,end=pred_len+1,step=1).flip(dims=(0,)).to(device)
    weights = weights/weights.sum()
    aggregated_nmse = torch.zeros(batch_size)
    for batch in range(batch_size):
        aggregated_nmse[batch] = torch.dot(nmse[batch], weights)
    # aggregated_nmse = torch.mean(torch.mean(nmse, dim=1), dim=0) # UNWEIGHTED
    aggregated_nmse = torch.mean(aggregated_nmse, dim=0)
    return aggregated_nmse





### RESERVOIR
# Define training setup
# criterion
criterion = NormalizedMeanSquaredError
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
print("Reservoir training...")
# start counting the time
start = time.time()
# Train the model
val_results, train_losses = (
    evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_dataloader, device, scheduler))
# stop counting the time
end = time.time()
print('Time elapsed: ', end - start, "s")
print(30*"-")





### BENCHMARK MODEL
print("Benchmark training...")
# training setup
# criterion
criterion = NormalizedMeanSquaredError
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
# start counting the time
start = time.time()
# Train the benchmark model
val_results_benchmark, train_losses_benchmark = (
    evaluate(num_epochs, criterion, optimizer, modelBenchmark, train_dataloader, val_dataloader, device, scheduler))
# stop counting the time
end = time.time()
print('Time elapsed: ', end - start, "s")
print(30*"-")





# Plotting the predictions
plt.figure(figsize=(15, 5))

stepToShow = min(pred_len, 100)
for i, var_name in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 4, i+1)
    plt.plot(val_t[stepToShow:], val_target_np[:, stepToShow - 1, i], label='True')
    plt.plot(val_t[stepToShow:], val_predictions_np[:, stepToShow - 1, i], label='Predicted (Reservoir)')
    plt.plot(val_t[stepToShow:], val_predictions_np_benchmark[:, stepToShow - 1, i], label='Predicted (Benchmark)')
    # plot the residuals between the predicted and the true values
    plt.plot(val_t[stepToShow:], val_target_np[:, stepToShow - 1, i] - val_predictions_np[:, stepToShow - 1, i], label='Residuals (Reservoir)')
    plt.xlabel('Time')
    plt.ylabel(var_name)
    plt.legend()

plt.tight_layout()
plt.savefig('Media/lorenz_predictions.png')
plt.show()
plt.close()

# plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('Media/lorenz_loss_reservoir.png')
plt.show()
plt.close()

# plot the loss
plt.plot(lossesBenchmark)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('Media/lorenz_loss.png')
plt.show()
plt.close()
