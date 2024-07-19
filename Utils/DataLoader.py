from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import platform

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
<<<<<<< HEAD
        data = df[['x','y', 'vx', 'vy']].values
=======
        data = df[['x','y']].values
>>>>>>> 81fcb785cf9c553b07a1a58a6ea9991231a8b94f
        data = data[::20]
        perc_init_fit=0.05
        perc_input_fit=0.7
        perc_init_gen=0.05
        perc_input_gen=0.7
        perc_gen=0.95-perc_input_gen-perc_init_gen
<<<<<<< HEAD
=======

    if dataset=="Damped_harmonic":
        t = torch.arange(0,100,1e-2)
        data = 10*torch.exp(-0.03*t)*torch.cos(t)
        data = data.numpy().reshape(-1,1)
        perc_init_fit=0.1
        perc_input_fit=0.5
        perc_init_gen=0.1
        perc_input_gen=0.5
        perc_gen=0.9-perc_input_gen-perc_init_gen
>>>>>>> 81fcb785cf9c553b07a1a58a6ea9991231a8b94f


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
    
    return (input_fit, target_fit), (input_gen, target_gen), scaler