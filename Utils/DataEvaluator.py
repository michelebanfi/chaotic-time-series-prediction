import torch

def evaluate(num_epochs, criterion, optimizer, model, train_dataloader, val_sequences_torch, val_targets_torch):
    losses = []

    for epoch in range(num_epochs):

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_sequences_torch[:, :, :])
            rmse = torch.sqrt(criterion(val_predictions, val_targets_torch))
            val_predictions_np = val_predictions.squeeze(0).numpy()
            val_target_np = val_targets_torch.squeeze(0).numpy()
            print(f'RMSE: {rmse.item():.4f}')

        model.train()

    return val_predictions_np, val_target_np, model, losses