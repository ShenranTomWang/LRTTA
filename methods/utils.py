import torch

def log_var(features):
    var = torch.var(features, dim=(0, 2, 3))
    var = torch.log(var)
    feat = var[None, ...]
    return feat

def mean(features):
    mean = features.mean(dim=[2, 3])
    return mean.mean(dim=0)[None, ...]

def var(features):
    var = torch.var(features, dim=(0, 2, 3))
    return var[None, ...]

def mean_var(features):
    # Compute channel-wise mean and log variance
    mean = features.mean(dim=[2, 3])  # [B, C]
    variance = features.var(dim=[2, 3]) + 1e-5  # [B, C], add epsilon for stability
    log_variance = torch.log(variance)
    
    # Concatenate mean and log variance to form the style vector
    style_vector = torch.cat([mean, log_variance], dim=1)  # [B, 2C]
    return style_vector.mean(dim=0)[None, ...]

def gram_matrix_diagonal(features):
    """
    Compute the diagonal of the Gram matrix.

    Args:
    - features: Tensor of shape [B, C, H, W] (Batch, Channels, Height, Width)

    Returns:
    - diagonal: Tensor of shape [B, C], the diagonal of the Gram matrix
    """
    B, C, H, W = features.shape
    features = features.view(B, C, -1)  # Flatten spatial dimensions: [B, C, H*W]
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram matrix: [B, C, C]
    diagonal = gram.diagonal(dim1=-2, dim2=-1)  # Extract diagonal elements: [B, C]
    return diagonal.mean(dim=0)[None, ...]