import torch

def evaluate(num_epochs, criterion, optimizer, currentModel, train_dataloader, val_dataloader, device):
    train_losses = []
    val_results = {'inputs':[], 'predictions':[], 'targets':[], 'losses':[]}

    for epoch in range(num_epochs):

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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.3f}')

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
                val_results['losses'].append(val_loss)

    return val_results, train_losses
