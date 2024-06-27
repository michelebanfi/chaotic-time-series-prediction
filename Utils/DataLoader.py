from sklearn.model_selection import train_test_split
import torch

# load data function
def loadData(data, t, pred_len, input_len, train_batch_size=1, val_batch_size=1):

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_t, val_t = train_test_split(t, test_size=0.2, shuffle=False)

    # Function to create inputs
    def create_sequences(data, pred_len, input_len):
        inputs = torch.zeros(size=(1, input_len, data.size(1)), dtype=torch.float32)
        targets = torch.zeros(size=(1, pred_len, data.size(1)), dtype=torch.float32)
        n_data = data.size(0)
        for start_input in range(0, n_data - pred_len, input_len + pred_len):
            if start_input + input_len + pred_len >= n_data: break # cut tail that do not fit the size of data

            new_input = data[start_input:start_input + input_len].unsqueeze(0)
            inputs = torch.cat((inputs, new_input), dim=0)

            new_target = data[start_input + input_len:start_input + pred_len + input_len].unsqueeze(0)
            targets = torch.cat((targets, new_target))

        # remove first entry -> zeros by initialization
        inputs = inputs[1:, :, :].float()
        targets = targets[1:, :, :].float()
        return inputs, targets

    # Create sequences for training and validation
    train_sequences, train_targets = create_sequences(train_data, pred_len, input_len)
    val_sequences, val_targets = create_sequences(val_data, pred_len, input_len)

    
    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(val_sequences, val_targets)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_t, train_dataloader, val_t, val_dataloader