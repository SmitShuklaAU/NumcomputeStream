import numpy as np
    
# ----------------------------------------------------------------------------
# StandardScaler
# ----------------------------------------------------------------------------

class StandardScaler():
    """
    Z-score standardization: z=(x-mean)/std
    
    Attributes
    ----------
    mean : np.ndarray, shape(n,)
    std  : np.ndarray, shape(n,) 
    
    Complexity
    ----------
    Time  : O(n*m)
    Space : O(n)
    """
    
    def __init__(self,eps:float=1e-12) -> None:
        self.mean_:np.ndarray|None= None
        self.std_ :np.ndarray|None= None
        self.eps:float=eps
    
    def fit(self, x:np.ndarray) -> "StandardScaler":
        """
        Compute mean and std from x.
        
        Parameters
        ----------
        x: np.ndarray, shape(n,m)
        
        Returns
        -------
        self
        
        Raises
        ------
        ValueError : if input array is empty & if array is not 2D
        """
        x=np.asarray(x, dtype=np.float64)
        if x.ndim==1:
          x=x.reshape(-1,1)  
        if x.ndim!=2:
            raise ValueError(f" Expected 2D array, got {x.ndim}D")
        if x.size==0:
            raise ValueError("Input array is empty")
        
        with np.errstate(all='ignore'):
            col_means=np.nanmean(x,axis=0)
            std=np.nanstd(x,axis=0)
       
        self.mean_=np.where(np.isnan(col_means),0.0,col_means)
       
        std=np.where(np.isnan(std),0.0,std)
        self.std_=np.where(std<self.eps,1.0,std)    # To avoid division by zero
        return self
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        """
        Apply Z-score Standardization

        Parameters
        ----------
        x: np.ndarray, shape(n,m)

        Returns
        -------
        z : np.ndarray, shape(n,m)
        
        Raises
        ------
        RuntimeError - if transform function is called before fit function
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        
        x=np.ascontiguousarray(np.asarray(x,dtype=np.float64))
        if x.ndim==1:
            x=x.reshape(-1,1)
        
        if x.shape[1] !=self.mean_.shape[0]:
            raise ValueError(f"expected {self.mean_.shape[0]} columns, got {x.shape[1]}")
        nan_mask=np.isnan(x)
        x=np.where(nan_mask,self.mean_,x)
        return (x-self.mean_)/self.std_
    def fit_transform(self,x:np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
class MinMaxScaler():
    """
    Scales each column to given [feature_min, feature_max] range.
    Default range is [0,1].

    Attributes
    ----------
    min_        : np.ndarray, shape(n,) - per column min seen at fit
    max_        : np.ndarray, shape(n,) - per column max seen at fit
    data_range_ : np.ndarray, shape(n,) - max_ - min_
    
    Raises
    ------
    ValueError : if feature is not between 0 and 1 
    
    Complexity
    ----------
    Time : O(n*m)
    Space: O(n)
    """
    def __init__(self,feature_range:tuple[int,int]=(0,1),eps:float=1e-12) -> None:
        if feature_range[0]>=feature_range[1]:
            raise ValueError("feature range must satisfy min<max")
        
        self.feature_range:tuple[int,int]=feature_range
        self.eps:float=eps
        self.min_:np.ndarray|None=None
        self.max_:np.ndarray|None=None
        self.data_range_:np.ndarray|None=None
        
    def fit(self,x:np.ndarray) -> "MinMaxScaler":
        """
        Compute per-column min and max from x.

        Parameters
        ----------
        X : np.ndarray, shape(n,m)
        
        Returns
        -------
        Self
        
        Raises
        ------
        ValueError : if input array is empty & if array is not 2D
        """
        x= np.asarray(x,dtype=np.float64)
        if x.ndim==1:
            x=x.reshape(-1,1)
        if x.ndim!=2:
            raise ValueError(f"Expected 2D array, got {x.ndim}D")
        if x.size==0:
            raise ValueError("Input array is empty")
        min_ = np.nanmin(x,axis=0)
        max_ = np.nanmax(x,axis=0)
        
        self.min_=min_
        self.max_=max_
        
        data_range=max_ - min_
        self.data_range_=np.where(data_range<self.eps,1.0,data_range)
        
        return self
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        """
        Scale x to feature range

        Parameters
        ----------
        x : np.ndarray, shape(n,m) 
        
        Returns
        -------
        z : np.ndarray, shape(n,m)
        
        Raises
        ------
        RuntimeError : if transform function is called before fit function
        """
        if self.min_ is None:
            raise RuntimeError("Call fit() before transform()")
    
        
        x=np.asarray(x,dtype=np.float64)
        if x.ndim==1:
            x=x.reshape(-1,1)
            
        if x.shape[1] !=self.min_.shape[0]:
            raise ValueError(f"expected {self.min_.shape[0]} columns, got {x.shape[1]}")
       
        f_min, f_max= self.feature_range
        x_std=(x-self.min_)/self.data_range_
        
        return x_std*(f_max-f_min)+f_min
    
    def fit_transform(self,x:np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
    
# ----------------------------------------------------------------------------
# OneHotEncoder
# ----------------------------------------------------------------------------

class OneHotEncoder():
    """
    Encodes categorical features

    Attributes
    ----------
    categories_ : list of np.ndarray - unique sorted values per column
    
    Raises
    ------
    ValueError : if handle_unknown is not 'ignore' or 'error'
    
    Complexity
    ----------
    Time  : O(n*m*k)
    Space : O(n*sum(k))
    """
    
    def __init__(self,handle_unknown:str="ignore") -> None:
        if handle_unknown not in ("ignore","error"):
           raise ValueError("handle unkown must be 'ignore' or 'error'")
        self.handle_unknown:str=handle_unknown
        self.categories_:list[np.ndarray]|None= None 
       
    def fit(self, x:np.ndarray) -> "OneHotEncoder":
        """
        Learn unique categories per column

        Parameters
        ----------
        x : np.ndarray, shape(n,m)

        Returns
        -------
        self
        
        Raises
        ------
        ValueError : if input array is empty
        """
        
        x=np.asarray(x)
        if x.ndim==1:
            x=x.reshape(-1,1)
        if x.size==0:
            raise ValueError("Input array is empty")    
        # Stores unique values of each column
        self.categories_=[np.unique(x[:,i]) for i in range(x.shape[1])]
        return self
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        """
        One-hot encode X

        Parameters
        ----------
        x : np.ndarray, shape(n,m)

        Returns
        -------
        z = np.ndarray, shape(n, sum of category counts per col) 
        
        Raises
        ------
        RuntimeError - if transform function is called before fit function
        """
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform()")
        x=np.asarray(x)
        if x.ndim==1:
            x=x.reshape(-1,1)
            
        # One loop oover column is required as each column has
        # is own category set of different size, so can not broadcast together
        encoded=[]
        for i,cats in enumerate(self.categories_):
            col=x[:,i]
            if self.handle_unknown=="error":
                unseen=~np.isin(col,cats)
                if unseen.any():
                    raise ValueError(f"Column {i} contains unknown categories.")
            encoded.append((col[:,None]==cats).astype(float))        
        return np.hstack(encoded)
    def fit_transform(self,x:np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)  
# ----------------------------------------------------------------------------
# SimpleImputer
# ----------------------------------------------------------------------------

class SimpleImputer():
    """
    Replaces NaN values with a constant.
    
    Parameters
    ----------
    strategy : str - 'constant','mean','median' (default = 'mean')
    constant : float - value used when strategy='constant' (default = 0)
    
    Raises
    ------
    ValueError : if strategy is not 'constant','mean' or 'median'
    
    Complexity
    ----------
    Time : O(n*m)
    Space : O(m)
    """
    
    def __init__(self,strategy:str="mean",constant:float=0.0) -> None:
        if strategy not in ("constant", "mean", "median"):
            raise ValueError("Strategy must be 'constant','mean' or 'median'")
        self.strategy:str=strategy
        self.constant:float=constant
        self.fill_values_:np.ndarray|None=None
    
    def fit(self,x:np.ndarray) -> "SimpleImputer":
        """
        fill values from x
        
        Parameters
        ----------
        x: np.ndarray, shape(n,m)

        Raises
        -------
        ValueError: if input array is empty & if array is not 2D

        Returns
        -------
        self : SimpleImputer
        """
        x=np.asarray(x,dtype=np.float64)
        if x.ndim==1:
            x=x.reshape(-1,1)
        if x.ndim!=2:
            raise ValueError(f"Expected 2D array, got {x.ndim}D")
        if x.size==0:
            raise ValueError("Input array is empty")
        if self.strategy=="mean":
            col_means=np.nanmean(x,axis=0)
            self.fill_values_=np.where(np.isnan(col_means),0.0,col_means)
        elif self.strategy=="median":
            with np.errstate(all='ignore'):
                median_vals=np.nanmedian(x,axis=0)
            self.fill_values_=np.where(np.isnan(median_vals),0,median_vals)
        else:
            self.fill_values_=np.full(x.shape[1],self.constant)
        return self
    
    def transform(self, x:np.ndarray) -> np.ndarray:
        """
        Replaces NaN values in x using fill values from fit function

        Parameters
        ----------
        x : np.ndarray, shape(n,m)
        
        Raises
        ------
        RuntimeError : if transform function is called before fit function

        Returns
        -------
        x_imputed : np.ndarray, shape(n,m)
        """
        if self.fill_values_ is None:
            raise RuntimeError("Call fit() before transform()")
        
       
        x=np.asarray(x,dtype=np.float64).copy()
        if x.ndim==1:
            x=x.reshape(-1,1)
        
        if x.shape[1] !=self.fill_values_.shape[0]:
            raise ValueError(f"expected {self.fill_values_.shape[0]} columns, got {x.shape[1]}")
        
        nan_mask=np.isnan(x)
        x=np.where(nan_mask,self.fill_values_,x)
        return x    
    
    def fit_transform(self,x:np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)