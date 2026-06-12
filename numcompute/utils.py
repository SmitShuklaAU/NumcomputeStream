"""
A general purpose numerical utilities file

Public APIs
-----------
Distances 
    Euclidean_distance(a,b)
    pairwise_distance(x,y,metric) 

Activations
-----------
    Sigmoid(x)
    softmax(x,axis)

Numerical
---------
    logsumexp(x,axis)
    
Top-K Helper
------------
    topk_indices(values,k)

Batching
--------
    make_batches(x,batch_size)
    batch_iterator(x, y, batch_size, shuffle)
"""
import numpy as np

# ----------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------

def euclidean_distance(a:np.ndarray,b:np.ndarray) -> float:
    """
    distance between 2 1-D vectors.

    Parameters
    ----------
    a,b : np.ndarray, shape(n,)

    Returns
    -------
    float

    Raises
    ------
    valueError : if shapes differ

    Complexity
    ----------
    O(n)
    """

    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    
    if a.shape!=b.shape:
        raise ValueError(f"Shape mismatch between a and b")
    return float(np.sqrt(np.sum((a-b)**2)))

def pairwise_distance(x:np.ndarray,y:np.ndarray,metric:str="euclidean"):
    """
    Compute pairwise distances between rows of x and y metrics
    
    Parameters
    ----------
    x : np.ndarray, shapee(n,d)
    y : np.ndarray, shape(m,d)
    metric : {'euclidean','manhattan','cosine'}, default 'euclidean'
    
    Returns
    -------
    np.ndarray, shape(n,m) 
    
    Raises
    ------
    ValueError : if metric is not known or dimensions do not match
    
    Complexity
    ----------
    Time  : O(n x m x d)
    Space : O(n x m)
    """
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:
        x=x.reshape(1,-1)
    if y is None:
        y=x
    else:
        y=np.asarray(y,dtype=np.float64)
        if y.ndim==1:
            y=y.reshape(1,-1)
    
    if x.shape[1]!=y.shape[1]:
        raise ValueError(f"Feature dimension mismatched in x and y")
    
    if metric=='euclidean':
        xx=np.sum(x**2,axis=1,keepdims=True)
        yy=np.sum(y**2,axis=1,keepdims=True).T
        D2=xx+yy-2*x@y.T
        D2=np.clip(D2,0,None)
        return np.sqrt(D2)
    
    elif metric=='manhattan':
        return np.sum(np.abs(x[:,None,:]-y[None,:,:]),axis=2)
    
    elif metric=='cosine':
        x_norm=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        y_norm=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
        
        cosine=x_norm @ y_norm.T
        return 1.0-cosine
    
    else:
        raise ValueError(f"Unknown metric, choose from 'euclidean','manhattan','cosine'")
    
# ----------------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------------

def sigmoid(x:np.ndarray) -> np.ndarray:
    """
    Element-wise logistic sigmoid.

    Parameters
    ----------
    x : np.ndarray
    
    Returns
    -------
    np.ndarray, same shape, values in (0,1)
    
    Complexity: O(n)
    """
    
    x=np.asarray(x,dtype=np.float64)
    
    out = np.empty_like(x)
    pos = x >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out

def softmax(x:np.ndarray,axis:int=-1) -> np.ndarray:
    """
    softmax along axis 

    Parameters
    ----------
    x: np.ndarray
    axis: int, default-1
    
    Returns
    -------
    np.ndarray, same shape, rows sum to 1 along axis
    
    Complexity: O(n)
    """
    x=np.asarray(x,dtype=np.float64)
    m=x-np.max(x,axis=axis,keepdims=True)
    e=np.exp(m)
    
    return e/np.sum(e,axis=axis, keepdims=True)

# ----------------------------------------------------------------------------
# Numerical
# ----------------------------------------------------------------------------

def logsumexp(x:np.ndarray,axis:int|None) -> np.ndarray:
    """
    compute log(sum(exp(x))) in a numerical way

    Parameters
    ----------
    x : np.ndarray
    axis : int (but can be None)
    
    Returns
    -------
    np.ndarray
    
    Complexity : O(n)
    """
    
    x=np.asarray(x,dtype=np.float64)
    c=np.max(x,axis=axis,keepdims=True)
    
    if axis is None:
        c = np.max(x)
        return c + np.log(np.sum(np.exp(x - c)))

    return np.squeeze(c,axis=axis)+np.log(
        np.sum(np.exp(x-c),axis=axis)
    )
    
# ----------------------------------------------------------------------------
# Top k 
# ----------------------------------------------------------------------------
def topk_indices(values:np.ndarray,k:int) -> np.ndarray:
    """
    Returns indices of the top-k largest value

    Parameters
    ----------
    values : np.ndarray
    k : int

    Returns
    -------
    np.ndarray of indices
    """
    if k < 1 or k > len(values):
        raise ValueError(f"k must be between 1 and {len(values)}")
    return np.argpartition(values,-k)[-k:]

# ----------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------

def make_batches(n:int,batch_size:int)-> list[tuple[int,int]]:
    """
    Return a list of index paies for batching n samples

    Parameters
    ----------
    n : total number of samples
    batch_size : required batch size
    
    Returns
    -------
    list of tuples
    
    Raises
    ------
    ValueError : if batch_size is < 1
    
    Complexity : O(n/batch_size)
    """
    
    if batch_size<1:
        raise ValueError(f"batch_size must be atleast 1")
    return [(i,min(i+batch_size,n)) for i in range(0,n,batch_size)]

def batch_iterator(x:np.ndarray, y:np.ndarray|None, batch_size:int=32, shuffle:bool=True):
    """
    Yield (x_batch) or (x_batch,y_batch) tuples for mini-batch iteration
    
    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    batch_size : int, default 32
    shuffle : bool, default True
    
    Yield
    -----
    give x_batch or x_batch,y_batch depending on if y is provided

    Raises
    ------
    ValueError : if x and y have different first dimensions
    
    Complexity
    ----------
    Time : O(n)
    Space : O(batch_size)
    """
    x = np.asarray(x)
    n = x.shape[0]
    if y is not None:
        y=np.asarray(y)
        if y.shape[0]!=n:
            raise ValueError("x and y should have same number of samples")
    
    if shuffle:
        idx=np.random.permutation(n)
    else:
        idx=np.arange(n)   
         
    for start,end in make_batches(n,batch_size):
        batch_idx=idx[start:end]
        if y is None:
            yield(x[batch_idx],)
        else:
            yield(x[batch_idx],y[batch_idx])
        