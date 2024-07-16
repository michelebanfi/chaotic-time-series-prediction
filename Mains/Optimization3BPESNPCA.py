import sys
import os
sys.path.append(os.getcwd())

from Hyperoptimization.hyperoptESNPCA import optimize
import torch
from Reservoirs.ESNPCA import ESNPCA
import matplotlib.pyplot as plt
from decimal import Decimal
import itertools
from Utils.DataLoader import loadData

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plot_train=False

file="R3BP"
version="0"
data_filename=f"{file}_{version}"

# load data with standard params
(input_fit, target_fit), (input_gen, target_gen) = loadData(file, version)
io_size = input_fit.size(1)
n_gen = target_gen.size(0)
n_input_gen = input_gen.size(0)

## PLOT DATA
plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
plt.plot(input_fit[:,0].cpu(), input_fit[:,1].cpu(), label="Input fit")
plt.subplot(2,1,2)
plt.plot(target_gen[:,0].cpu(), target_gen[:,1].cpu(), label="Target generation")
plt.legend()
plt.show()


models_structures = {
    'reservoir_size':[256,512,1024,2048],
    'components':[0.005,0.01,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.1,0.15,-1]
}
keys, values = zip(*models_structures.items())
structures = [dict(zip(keys, v)) for v in itertools.product(*values)]

for i, structure in enumerate(structures):

    # search space
    search_space = {
        'components':[structure['components']], 
        'reservoir_size':[structure['reservoir_size']],
        'spectral_radius':[0.6,0.9,1.2,1.5],
        'leaking_rate':[0.1,0.3,0.5,0.7,0.9,1],
        'sparsity':[0.1,0.3],  
        'warmup':[100],
        'seed':[0]
    }

    model, args, loss, (pred_fit, upd_target_fit), pred_gen = optimize(model_class=ESNPCA, 
                                input_size=io_size, output_size=io_size, 
                                data_train=(input_fit, target_fit), data_generation=None,
                                search_full_space=False, nextractions=100, ntests=1,
                                device=device, verbose=False,
                                model_savepath=None, 
                                **search_space)
    
    args_str = ""
    for key in args:
        args_str += f"{key}={args[key]}_"

    print(f"Loss: {Decimal(loss):.2E} - Parameters: {args}")
    torch.save(model.state_dict(), f"Models/R3BP/ESNPCA_{data_filename}_optimization/best_model_{args_str}.pth")

    try:
        pred_gen = model.generate(input_gen, n_gen)
    except Exception as e:
        print(f"Exception in generation with {args}")
        break
    
    ## PLOTS
    # FIT
    if plot_train:
        plt.figure(figsize=(15,15))
        plt.title("Training")
        for v in range(io_size):
            plt.subplot(io_size,1,v+1)
            plt.plot(upd_target_fit[:,v].cpu(), label="Target", linestyle="--")
            plt.plot(pred_fit[:,v].cpu(), label="Predicted")
        plt.legend()
        plt.savefig(f"Media/R3BP/ESNPCA_{data_filename}_optimization/ESNPCA_fitting_{args_str}.png")
        plt.close()



    # GENERATION
    plt.figure(figsize=(15,15))
    plt.title("Generation")
    for v in range(io_size):
        plt.subplot(io_size,1,v+1)
        plt.plot(range(n_input_gen), input_gen[:,v].cpu(), label="Input")
        plt.plot(range(n_input_gen, n_input_gen+n_gen), target_gen[:,v].cpu(), label="Target", linestyle="--")
        plt.plot(range(n_input_gen, n_input_gen+n_gen), pred_gen[:,v].cpu(), label="Predicted")
    plt.legend()
    plt.savefig(f"Media/R3BP/ESNPCA_{data_filename}_optimization/ESNPCA_generation_{args_str}.png")
    plt.close()

    # 2D GENERATION
    plt.figure(figsize=(15,15))
    plt.plot(target_gen[:,0].cpu(), target_gen[:,1].cpu(), label="True")
    plt.plot(pred_gen[:,0].cpu(), pred_gen[:,1].cpu(), label="Generated")
    plt.legend()
    plt.savefig(f"Media/R3BP/ESNPCA_{data_filename}_optimization/ESNPCA_3Dgeneration_{args_str}.png")
    plt.close()


#############
## RESULTS ##
#############
# comparing the generation results
# the best amount of components is roughly 0.05 the reservoir size7
# the bigger the reservoir the better the prediction
# for res_size = 256 spectral_radius = 0.9, leaking_rate=0.7, sparsity=0.3
# for res_size = {512,1024,2048} spectral_radius = 0.9, leaking_rate=0.7, sparsity=0.3