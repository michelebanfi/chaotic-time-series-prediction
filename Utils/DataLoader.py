from sklearn.model_selection import train_test_split
import numpy as np
import torch

# load data function
def loadData(data, t, seq_len, batch_size=1):

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_t, val_t = train_test_split(t, test_size=0.2, shuffle=False)

    # Function to create sequences
    def create_sequences(data, seq_len):
        sequences = []
        targets = []
        for i in range(len(data) - seq_len):
            sequences.append(data[i:i + 1])
            targets.append(data[i + 1:i + seq_len + 1])
        return np.array(sequences), np.array(targets)

    # Create sequences for training and validation
    train_sequences, train_targets = create_sequences(train_data, seq_len)
    val_sequences, val_targets = create_sequences(val_data, seq_len)

    # Convert to PyTorch tensors
    train_sequences_torch = torch.tensor(train_sequences, dtype=torch.float32)
    train_targets_torch = torch.tensor(train_targets, dtype=torch.float32)
    val_sequences_torch = torch.tensor(val_sequences, dtype=torch.float32)
    val_targets_torch = torch.tensor(val_targets, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(train_sequences_torch, train_targets_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_sequences_torch, val_targets_torch, val_t