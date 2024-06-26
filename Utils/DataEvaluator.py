import torch

def NormalizedMeanSquaredError(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2) #/ torch.mean(y_true ** 2)

def evaluate(num_epochs, criterion, optimizer, currentModel, train_dataloader, val_sequences_torch, val_targets_torch):
    losses = []

    for epoch in range(num_epochs):

        currentModel.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = currentModel(inputs)
            targets = targets[-1, :, :]
            outputs = outputs.squeeze(0)
            loss = NormalizedMeanSquaredError(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        currentModel.eval()
        with torch.no_grad():
            val_predictions = torch.zeros(val_targets_torch.size(0), val_targets_torch.size(1), val_sequences_torch.size(2))
            for i in range(val_sequences_torch.size(0)):
                val_predictions[i,:,:] = currentModel(val_sequences_torch[i, :, :].unsqueeze(0))
            #rmse = torch.sqrt(criterion(val_predictions, val_targets_torch))
            val_predictions_np = val_predictions.squeeze(0).numpy()
            val_target_np = val_targets_torch.squeeze(0).numpy()
            #print(f'RMSE: {rmse.item():.4f}')

    return val_predictions_np, val_target_np, losses
