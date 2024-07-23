import torch

# loss weighted as criterion
def NormalizedMeanSquaredError(y_pred, y_true):
    ndim = y_true.ndim
    squared_dist = torch.sum((y_true - y_pred)** 2, dim=ndim-1)
    true_squared_norm = torch.sum(y_true ** 2, dim=ndim-1)
    loss = squared_dist / true_squared_norm
    loss = torch.mean(loss, dim=ndim-2)
    if ndim == 3:
        loss = torch.mean(loss, dim=0) 
    return loss

def VarianceNormalizedSquaredError(y_pred, y_true):
    ndim = y_true.ndim
    dist = torch.norm((y_true - y_pred)**2)
    var_pred = torch.var(y_pred, dim=ndim-2, keepdim=True)
    loss = dist / var_pred
    
    loss = torch.mean(loss, dim=ndim-2)
    loss = torch.mean(loss, dim=-1)
    
    if ndim == 3:
        loss = torch.mean(loss, dim=0) 

    return loss