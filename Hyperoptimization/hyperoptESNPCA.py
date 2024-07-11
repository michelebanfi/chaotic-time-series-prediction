from torch import nn
from Utils.Losses import NormalizedMeanSquaredError
import torch
from tqdm import tqdm



criterion = NormalizedMeanSquaredError

def optimize(model_class:nn.Module, input_size, reservoir_size, output_size, data_train, model_savepath, data_generation=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), nextractions=10, ntests=1, **kwargs):
    best_loss = torch.inf
    best_model, best_args, best_train_pred, best_gen_pred = None, None, None, None
    # keep track of already sampled args
    selected_args = []

    
    # data = (input, target)
    input_train, target_train = data_train[0].to(device), data_train[1].to(device)
    input_gen, target_gen = data_generation[0].to(device), data_generation[1].to(device)
    ndim = input_gen.ndim

    for _ in tqdm(range(nextractions)):
        # randomly sample hyperparameters
        args = extract_args(**kwargs)
        # if previously selected, extract again
        while(args in selected_args):
            args = extract_args(**kwargs)
        selected_args.append(args)

        for _ in range(ntests):
            # instantiate model
            model=model_class(input_size, reservoir_size, output_size, **args).to(device)


            ## FIT
            try:
                pred_train = model.fit(input_train, target_train)
            except Exception as e:
                tqdm.write("Exception in training")
                break
            # custom loss
            loss_train = criterion(pred_train, target_train, ndim).item()


            ## GENERATION
            if data_generation is not None:
                try:
                    pred_gen = model.generate(input_gen, target_gen.size(0))
                except Exception as e:
                    tqdm.write("Exception in generation")
                    break

                # custom loss
                loss_generation = criterion(pred_gen, target_gen, ndim).item()

                # if generation is available then use it to validate
                if loss_generation < best_loss:
                    tqdm.write(f"Best generation loss: {loss_generation:.4f} with {args}")
                    best_model = model
                    best_args = args
                    best_loss = loss_generation
                    best_gen_pred = pred_gen
                    best_train_pred = pred_train
                    # save model in case of errors
                    torch.save(best_model.state_dict(), model_savepath)
            else: 
                # if generation data are not available use training
                if loss_train < best_loss:
                    tqdm.write(f"Best training loss: {loss_train:.4f} with {args}")
                    best_model = model
                    best_args = args
                    best_loss = loss_train
                    best_train_pred = pred_train
                    # save model in case of errors
                    torch.save(best_model.state_dict(), model_savepath)

    return best_model, best_args, best_loss, best_train_pred, best_gen_pred




def extract_args(**kwargs):
    args={}
    for key in kwargs:
        idx = torch.randint(0, len(kwargs[key]), size=(1,))
        args[key] = kwargs[key][idx]
    return args