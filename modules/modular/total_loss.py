def calculate_total_loss(supervised_loss, unsupervised_loss, lambda_sup):
    """
    Calculate the total loss by combining supervised and unsupervised losses
    
    Args:
        supervised_loss: The supervised branch loss
        unsupervised_loss: The unsupervised branch loss
        lambda_sup: Weighting factor for supervised loss
        
    Returns:
        Combined total loss
    """
    return unsupervised_loss + lambda_sup * supervised_loss