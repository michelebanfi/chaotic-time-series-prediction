from Utils.Losses import NormalizedMeanSquaredError as NMSE, VarianceNormalizedSquaredError as VNSE
import torch
from tqdm import tqdm
from decimal import Decimal
import itertools

criterion = NMSE

def optimize(model_class, input_size, output_size, data_train, model_savepath=None, data_generation=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), verbose=True, search_full_space=False, nextractions=10, ntests=1, **kwargs):
    # init params
    best_loss = torch.inf
    best_model, best_args, best_train_pred, best_gen_pred = None, None, None, None

    # unpack search space
    keys, values = zip(*kwargs.items())
    search_space_points = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # look in the full search space?
    if search_full_space:
        nextractions = len(search_space_points)

    # extract data
    # data = (input, target)
    input_train, target_train = data_train[0].to(device), data_train[1].to(device)
    if data_generation is not None:
        input_gen, target_gen = data_generation[0].to(device), data_generation[1].to(device)


    for _ in tqdm(range(nextractions)):
        # no more points -> interrupt
        if len(search_space_points) == 0:
            tqdm.write("Full space analysed")
            break
        # randomly sample hyperparameters
        idx_config = torch.randint(0, len(search_space_points), size=(1,)).item()
        args = search_space_points[idx_config]
        # remove from available params
        search_space_points.remove(args)

        # initialize the total loss of the configuration selected
        avg_loss_fit = 0
        avg_loss_gen = 0
        for _ in range(ntests):
            # instantiate model
            
            model=model_class(input_size, output_size, **args).to(device)
            
            ## FIT
            try:
                pred_train, upd_target_train = model.fit(input_train, target_train)
            except Exception as e:
                tqdm.write("Exception in training")
                break
            # custom loss
            loss_train = criterion(pred_train, upd_target_train).item()
            avg_loss_fit += loss_train


            ## GENERATION
            if data_generation is not None:
                try:
                    pred_gen = model.generate(input_gen, target_gen.size(0))
                except Exception as e:
                    tqdm.write("Exception in generation")
                    break

                # custom loss
                loss_generation = criterion(pred_gen, target_gen).item()

                # if generation is available use it to validate
                avg_loss_gen += loss_generation

        
        ## END OF ITERATION
        # save best configuration
        # once finished all the tests for the same params
        # the final goodness of the configuration is the average on the tests
        avg_loss_fit /= ntests
        avg_loss_gen /= ntests
        if data_generation is not None:
            avg_loss = avg_loss_gen
        else:
            avg_loss = avg_loss_fit
        # if generation is available then use it to validate
        if avg_loss < best_loss:
            if verbose:
                tqdm.write("")
                tqdm.write("BEST MODEL")
                tqdm.write(f"Parameters: {args}")
                tqdm.write(f"Training loss: {Decimal(avg_loss_fit):.2E}")
            best_model = model
            best_args = args
            best_loss = avg_loss
            best_train_pred = pred_train
            if data_generation is not None:
                if verbose: 
                    tqdm.write(f"Generation loss: {Decimal(avg_loss_gen):.2E}")
                best_gen_pred = pred_gen
            # save model in case of errors
            if model_savepath is not None:
                torch.save(best_model.state_dict(), model_savepath)

    return best_model, best_args, best_loss, (best_train_pred, upd_target_train), best_gen_pred