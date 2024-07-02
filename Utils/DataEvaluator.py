import torch
import numpy as np

def evaluate(num_epochs, criterion, optimizer, currentModel, train_dataloader, val_dataloader, device, scheduler=None):
    train_losses = []
    val_best_loss = np.inf
    val_best_results = {'inputs':[], 'predictions':[], 'targets':[], 'losses':[]}
    max_patience = 6
    patience = max_patience

    for epoch in range(num_epochs):
        ## begin of epoch
        print("")
        print(5*">", "New epoch", 5*"<")
        running_loss = []
        val_results = {'inputs':[], 'predictions':[], 'targets':[], 'losses':[]}
        patience-=1

        ## epoch
        # Train the model
        currentModel.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = currentModel(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            running_loss.append(loss.item())

        # Evaluate the model
        currentModel.eval()
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_predictions = currentModel(val_inputs)
                val_loss = criterion(val_predictions, val_targets)
                val_results['inputs'].append(val_inputs)
                val_results['predictions'].append(val_predictions)
                val_results['targets'].append(val_targets)
                val_results['losses'].append(val_loss.cpu().item())

            val_mean_loss = np.mean(val_results["losses"])
            if val_best_loss - val_mean_loss > 1e-4: # if the best model is sensibly worst then the current
                # save best model results
                val_best_loss = val_mean_loss
                val_best_results = val_results 
                # restore patience
                patience = max_patience
                print("!!! BEST MODEL !!!")

        ## end of epoch  
        # show info
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(running_loss):.3f}, Validation loss: {val_mean_loss:.3f}')
        # scheduler
        if scheduler is not None:
            print("Learning rate: %.5f" % scheduler.get_last_lr()[0])
            scheduler.step() 
        # patience  
        if patience == 0:
            print("Ran out of patience")
            break

    return val_best_results, train_losses
