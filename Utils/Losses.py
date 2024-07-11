import torch
# NMSE weighted as criterion
def NormalizedMeanSquaredError(y_pred, y_true, ndim=3):
    squared_dist = torch.sum((y_true - y_pred)** 2, dim=ndim-1) # squared euclidean distances between predictions
    true_squared_norm = torch.sum(y_true ** 2, dim=ndim-1) # to normalize the loss
    nmse = squared_dist / true_squared_norm
    # UNWEIGHTED
    aggregated_nmse = torch.mean(nmse, dim=ndim-2)
    if ndim == 3:
        aggregated_nmse = torch.mean(aggregated_nmse, dim=0) 
    return aggregated_nmse