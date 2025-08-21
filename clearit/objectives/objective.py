# clearit/objectives/objective.py
import torch

def compute_loss(lossfunc_name, sim_matrix, labels_matrix, batch_size, tau):
    """
    compute_loss(lossfunc_name, sim_matrix, labels_matrix, batch_size, tau)
    
    Computes the loss for a batch of data using the specified loss function.
    
    Parameters
    ----------
    lossfunc_name : str
        The name of the loss function to be used.
        "ntxent" for NT-Xent,
        "ntxentNC" for Non-contrastive NT-Xent (i.e. alignment loss only),
        "nce" for NCE loss
    sim_matrix : torch.Tensor
        The similarity matrix of the batch.
    labels_matrix : torch.Tensor
        The labels matrix of the batch.
    batch_size : int
        The size of the batch.
    tau : float
        The temperature parameter.
        
    Returns
    -------
    loss : torch.Tensor
        The loss of the batch.
    """
    
    if lossfunc_name == "ntxent":
        loss = ntxent_loss(sim_matrix, labels_matrix, batch_size, tau)
    elif lossfunc_name == "ntxentNC":
        loss = ntxentNC_loss(sim_matrix, labels_matrix, batch_size, tau)
    elif lossfunc_name == "nce":
        loss = nce_loss(sim_matrix, labels_matrix, batch_size, tau) 
    return loss

def nce_loss(sim_matrix, labels_matrix, batch_size, tau=1):
    """
    nce_loss(sim_matrix, labels_matrix, batch_size, tau=1)

    Computes the Noise Contrastive Estimation (NCE) loss for a batch of samples.
    
    Parameters
    ----------
    sim_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the similarity values of all pairs of images in the batch.
    labels_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the labels of all pairs of images in the batch.
    batch_size : int
        The batch size.
    tau : float, optional
        The temperature parameter.
    
    Returns
    -------
    loss : torch.Tensor
        The NCE loss for the batch.
    """
    diag = torch.eye(2 * batch_size).bool().cuda() # create matrix of size(2*batch_size) with True on its diagonal
    
    sim_matrix[diag] = 0    # use diag to set diagonal elements of
    labels_matrix[diag] = 0 # sim_matrix and labels_matrix to 0/False
    
    sim_matrix = torch.exp(sim_matrix/tau) # compute exponent of sim_matrix (diagonal is now 1 again)
    num = sim_matrix[labels_matrix.bool()] # compute numerator vector by taking all similarity values of positive pairs
    denom = torch.sum(sim_matrix, dim=0) # compute denominator vector by summing all similarity values per sample (similarity of sample x with itself + similarity of sample x with all other samples)
    loss = -torch.log(num/(denom+1e-10)) # compute loss vector by taking the negative logarithm
                                 # numerator should be as large as possible (maximum similarity of positive pairs)
                                 # denominator should be as small as possible (minimum similarity of negative pairs)
                                 # such that the negative logarithm becomes as small as possible (lower bound is 0 when all negative pair similarities are 0)
    
    return torch.sum(loss)/batch_size


import torch

def ntxent_loss(sim_matrix, labels_matrix, batch_size, tau=1, epsilon=1e-7):
    """
    ntxent_loss(sim_matrix, labels_matrix, batch_size, tau=1)

    Computes the NT-Xent loss for a batch of images.

    Parameters
    ----------
    sim_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the similarity values of all pairs of images in the batch.
    labels_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the labels of all pairs of images in the batch.
    batch_size : int
        The batch size.

    tau : float, optional
        The temperature parameter.

    Returns
    -------
    loss : torch.Tensor
        The NT-Xent loss for the batch.
    """
    diag = torch.eye(2 * batch_size).bool().cuda() # create matrix of size(2*batch_size) with True on its diagonal

    labels_matrix[diag] = 0 # sim_matrix and labels_matrix to 0/False
    sim_matrix = sim_matrix / tau # scale the similarity matrix
    
    num = sim_matrix[labels_matrix.bool()] # compute numerator vector by taking all similarity values of positive pairs
    
    # Exclude diagonal elements for the denominator computation
    sim_matrix[diag] = float('-inf') # replace diagonal with -inf for logsumexp to ignore these elements

    # Compute logsumexp across each row
    denom = torch.logsumexp(sim_matrix, dim=1) # compute denominator in a numerically stable way
    
    values = (num + epsilon) - denom # Using log properties: log(a/b) = log(a) - log(b)
    
    loss = -values # negative because we want to minimize the loss
    
    # Debugging checks
    if torch.isnan(loss).any():
        print("NaN values found in the computed loss!")
    if torch.isinf(loss).any():
        print("Inf values found in the computed loss!")
        
    return torch.sum(loss) / batch_size


def ntxent_loss_old(sim_matrix, labels_matrix, batch_size, tau=1, epsilon=1e-7):
    """
    ntxent_loss(sim_matrix, labels_matrix, batch_size, tau=1)
    
    Computes the NT-Xent loss for a batch of images.
    
    Parameters
    ----------
    sim_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the similarity values of all pairs of images in the batch.
    labels_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the labels of all pairs of images in the batch.
    batch_size : int
        The batch size.

    tau : float, optional
        The temperature parameter.

    Returns
    -------
    loss : torch.Tensor
        The NT-Xent loss for the batch.

        
    """
    diag = torch.eye(2 * batch_size).bool().cuda() # create matrix of size(2*batch_size) with True on its diagonal
    
    labels_matrix[diag] = 0 # sim_matrix and labels_matrix to 0/False
    sim_matrix = torch.exp(sim_matrix/tau) # compute exponent of sim_matrix (diagonal is now 1 again)
    num = sim_matrix[labels_matrix.bool()] # compute numerator vector by taking all similarity values of positive pairs
    sim_matrix2 = sim_matrix.clone()

    sim_matrix2[diag] = 0 # set diagonal entries to 0 again to exclude them in denominator sum
    denom = torch.sum(sim_matrix2, dim=0) # compute denominator vector by summing all similarity values per sample, excluding identical pair (only similarity of sample x with all other samples)
    
    values = ((num+epsilon)/(denom+epsilon))
    if (values <= 0).any():
        print("Found non-positive values for log input:", values[values <= 0])

    loss = -torch.log((num+epsilon)/(denom+epsilon)) # compute loss vector by taking the negative logarithm
                                 # numerator should be as large as possible (maximum similarity of positive pairs)
                                 # denominator should be as small as possible (minimum similarity of negative pairs)
                                 # such that the negative logarithm becomes as small as possible (lower bound is 0 when all negative pair similarities are 0)
    if torch.isnan(loss).any():
        print("NaN values found in the computed loss!")
    if torch.isinf(loss).any():
        print("Inf values found in the computed loss!")
        
    return torch.sum(loss)/batch_size

def ntxentNC_loss(sim_matrix, labels_matrix, batch_size, tau=1):
    """
    ntxentNC_loss(sim_matrix, labels_matrix, batch_size, tau=1)
    
    Computes only the alignment loss of the NT-Xent loss function for a batch of images.
    
    Parameters
    ----------
    sim_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the similarity values of all pairs of images in the batch.
    labels_matrix : torch.Tensor
        A matrix of size (2*batch_size, 2*batch_size) containing the labels of all pairs of images in the batch.
    batch_size : int
        The batch size.
    tau : float, optional
        The temperature parameter.

    Returns
    -------
    loss : torch.Tensor
        The alignment loss for the batch.

        
    """
    diag = torch.eye(2 * batch_size).bool().cuda() # create matrix of size(2*batch_size) with True on its diagonal
    
    sim_matrix[diag] = 0    # use diag to set diagonal elements of
    labels_matrix[diag] = 0 # sim_matrix and labels_matrix to 0/False
    
    sim_matrix = torch.exp(sim_matrix/tau) # compute exponent of sim_matrix (diagonal is now 1 again)
    num = sim_matrix[labels_matrix.bool()] # compute numerator vector by taking all similarity values of positive pairs
    denom = 1 # set the denominator to 1 in order to ignore distribution loss
    loss = -torch.log(num/(denom+1e-10)) # compute loss vector by taking the negative logarithm
                                 # numerator should be as large as possible (maximum similarity of positive pairs)
                                 # denominator should be as small as possible (minimum similarity of negative pairs)
                                 # such that the negative logarithm becomes as small as possible (lower bound is 0 when all negative pair similarities are 0)

    return torch.sum(loss)/batch_size



def loss(sim_matrix, labels_matrix, batch_size, tau=1, epsilon=1e-7):

    diag = torch.eye(2 * batch_size).bool().cuda()
    
    labels_matrix[diag] = 0
    sim_matrix = torch.exp(sim_matrix/tau)
    num = sim_matrix[labels_matrix.bool()]
    sim_matrix2 = sim_matrix.clone()

    sim_matrix2[diag] = 0
    denom = torch.sum(sim_matrix2, dim=0)
    
    values = ((num+epsilon)/(denom+epsilon))
    if (values <= 0).any():
        print("Found non-positive values for log input:", values[values <= 0])

    loss = -torch.log((num+epsilon)/(denom+epsilon))
    if torch.isnan(loss).any():
        print("NaN values found in the computed loss!")
    if torch.isinf(loss).any():
        print("Inf values found in the computed loss!")
        
    return torch.sum(loss)/batch_size