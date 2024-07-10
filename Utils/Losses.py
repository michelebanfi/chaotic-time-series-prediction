import torch
# NMSE weighted as criterion
def NormalizedMeanSquaredError(y_pred, y_true):
    device = y_pred.get_device()
    if device == -1:
        device = 'cpu'
    pred_len = y_pred.size(1)
    batch_size = y_pred.size(0)

    squared_dist = torch.sum((y_true - y_pred)** 2, dim=2) # squared euclidean distances between predictions
    true_squared_norm = torch.sum(y_true ** 2, dim=2)
    nmse = squared_dist / true_squared_norm
    # actual (from above) shape: (batch size, prediction length)
    # WEIGHTED
    #base = torch.tensor(1, dtype=torch.float32)
    #weights = base.pow(-torch.arange(start=1,end=pred_len+1,step=1)).to(device)
    #weights = weights/weights.sum()
    #aggregated_nmse = torch.zeros(batch_size)
    #for batch in range(batch_size):
    #    aggregated_nmse[batch] = torch.dot(nmse[batch], weights)
    # UNWEIGHTED
    aggregated_nmse = torch.mean(torch.mean(nmse, dim=1), dim=0) 
    aggregated_nmse = torch.mean(aggregated_nmse, dim=0)
    return aggregated_nmse