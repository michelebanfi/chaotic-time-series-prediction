from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import platform

diego = True
# load data function
def __loadData(pred_len, input_len, train_batch_size=1, val_batch_size=1, file="3BP", train_samples=100, val_samples=100, sampling_rate=10):
    num_files = 10
    
    if file == "3BP":
        variables = ['x', 'y']
    elif file == "lorenz":
        variables = ['x', 'y', 'z']

    dimensionality = len(variables)

    train_sequences = torch.zeros(size=(1, input_len, dimensionality), dtype=torch.float32)
    train_targets = torch.zeros(size=(1, pred_len, dimensionality), dtype=torch.float32)
    val_sequences = torch.zeros(size=(1, input_len, dimensionality), dtype=torch.float32)
    val_targets = torch.zeros(size=(1, pred_len, dimensionality), dtype=torch.float32)


    for i in range(0, num_files):
        if diego:
            df = pd.read_csv(f"D:/File_vari/Scuola/Universita/Bicocca/Magistrale/AI4ST/23-24/II_semester/AIModels/3_Body_Problem/Lorenz/Data/{file}_{i}.csv")
        else:
            df = pd.read_csv(f"Data/{file}_{i}.csv")

        data = torch.tensor(df[variables].values)
        t = df['time'].values

        data = data[::sampling_rate]
        t = t[::sampling_rate]

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
        train_t, val_t = train_test_split(t, test_size=0.2, shuffle=False)

        # Function to create inputs
        def create_sequences(data, pred_len, input_len, n_samples=100): # data: (points, dimension)
            inputs = torch.zeros(size=(1, input_len, data.size(1)), dtype=torch.float32) # to concat
            targets = torch.zeros(size=(1, pred_len, data.size(1)), dtype=torch.float32) # to concat

            # generate n starting points to sample n subsequences
            n_data = data.size(0)
            perm = torch.randperm(n_data - pred_len - input_len)
            starting_points = perm[:n_samples]
            for start_input in starting_points: #range(0, n_data - pred_len, input_len + pred_len):
                #if start_input + input_len + pred_len >= n_data: break # cut tail that do not fit the size of data
                

                new_input = data[start_input:start_input + input_len].unsqueeze(0)
                inputs = torch.cat((inputs, new_input), dim=0)

                new_target = data[start_input + input_len:start_input + pred_len + input_len].unsqueeze(0)
                targets = torch.cat((targets, new_target))

            # remove first entry -> zeros by initialization
            inputs = inputs[1:, :, :].float()
            targets = targets[1:, :, :].float()
            return inputs, targets

        # Create sequences for training and validation
        train_seq, train_tar = create_sequences(train_data, pred_len, input_len, n_samples=train_samples)
        val_seq, val_tar = create_sequences(val_data, pred_len, input_len, n_samples=val_samples)

        # Concatenate the sequences
        train_sequences = torch.cat((train_sequences, train_seq), dim=0)
        train_targets = torch.cat((train_targets, train_tar), dim=0)
        val_sequences = torch.cat((val_sequences, val_seq), dim=0)
        val_targets = torch.cat((val_targets, val_tar), dim=0)

    # remove first entry -> zeros by initialization
    train_sequences = train_sequences[1:, :, :]
    train_targets = train_targets[1:, :, :]
    val_sequences = val_sequences[1:, :, :]
    val_targets = val_targets[1:, :, :]

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    val_dataset = torch.utils.data.TensorDataset(val_sequences, val_targets)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_t, train_dataloader, val_t, val_dataloader

def loadData(dataset="R3BP", version="0", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    if platform.platform() == 'macOS-10.16-x86_64-i386-64bit':
        path = "../"
    else:
        path = ""

    if dataset=="lorenz":
        ## WHOLE DATA
        data_filename = f"lorenz_{version}"
        df = pd.read_csv(f"{path}Data/Lorenz/{data_filename}.csv")
        data = df[['x','y','z']].values
        data = data[::20]
        perc_init_fit=0.1
        perc_input_fit=0.5
        perc_init_gen=0.1
        perc_input_gen=0.5
        perc_gen=0.9-perc_input_gen-perc_init_gen

    if dataset=="R3BP":
        ## WHOLE DATA
        data_filename = f"3BP_{version}"
        df = pd.read_csv(f"{path}Data/R3BP/{data_filename}.csv")
        data = df[['x','y']].values
        data = data[::20]
        perc_init_fit=0.1
        perc_input_fit=0.5
        perc_init_gen=0.1
        perc_input_gen=0.5
        perc_gen=0.9-perc_input_gen-perc_init_gen

    # scale data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data).float().to(device)
    n_samples = data.size(0)

    # FIT DATA
    n_init_fit=int(n_samples*perc_init_fit)
    n_input_fit=int(n_samples*perc_input_fit)
    input_fit = data[n_init_fit:n_init_fit+n_input_fit]
    target_fit = data[n_init_fit+1:n_init_fit+n_input_fit+1]

    ## GENERATION DATA
    n_init_gen=int(n_samples*perc_init_gen)
    n_input_gen=int(n_samples*perc_input_gen)
    n_gen=int(n_samples*perc_gen)
    input_gen = data[n_init_gen:n_init_gen+n_input_gen]
    target_gen = data[n_init_gen+n_input_gen:n_init_gen+n_input_gen+n_gen] 
    
    return (input_fit, target_fit), (input_gen, target_gen)