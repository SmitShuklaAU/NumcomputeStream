
import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------------
# Core Function
# ----------------------------------------------------------------------------

def load_csv(path: str | Path ,
    delimiter: str=",",
    fill_value: float= np.nan,
    skip_header:int = 0,
    usecols:tuple[int,...]|list[int]|None= None,
    missing_strategy:str="fill") -> np.ndarray:
    """
    Loads a delimited file into Numpy array using np.genfromtxt.
    Missing values are filled with column mean.

    Parameters
    ----------
    path : str | Path             - path to the CSV file
    delimiter : str               - column separator, default ','
    fill_value : float            - placeholder for missing value, default np.nan
    skip_header : int             - number of header lines to skip, default is 0 
    usecols : sequence of integer - column indices to lead; None loads all data
    missing_strategy : str        - 'fill' replaces Nan with column mean (default)
                                    'skip' drops any row containing a NaN 

    Notes
    -----
    Columns containing only non-numeric values are automatically dropped.
    
    Returns
    -------
    data : np.ndarray, shape (n*m) 

    Raises
    ------
    FileNotFoundError - raises if file path does not exist
    ValueError        - if there is no data in file
    TypeError         - if path is not string or Path

    Complexity
    ----------
    Time : O(n*m)
    Space : O(n*m)
    """

    if missing_strategy not in ("fill","skip"):
        raise ValueError(f"Missing strategy must be 'fill' or 'skip', got '{missing_strategy}'")
    if not isinstance(path,(str,Path)):
        raise TypeError(f"File path must be string or Path, but got {type(path)}")
    
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not Found : {path}")

    if path.stat().st_size==0:
        raise ValueError(f"file is empty.")
    
    raw_data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header,
        filling_values=fill_value,
        encoding="utf-8",
        usecols=usecols,
        dtype=np.float64,
    )

    if raw_data.size==0:
        raise ValueError(f"No data rows found in {path.stem}")
    if raw_data.ndim ==1:
        raw_data=raw_data.reshape(1,-1)
    
    all_nan_cols = np.all(np.isnan(raw_data), axis=0)
    if np.any(all_nan_cols):
        text_col_indices = np.where(all_nan_cols)[0]
        
        raw_data = raw_data[:, ~all_nan_cols]
        
    if missing_strategy=='skip':
        row_mask=~np.isnan(raw_data).any(axis=1)
        data=raw_data[row_mask]
        if data.size==0:
            raise ValueError("All rows were dropped due to missing values")
    else:
        col_means=np.nanmean(raw_data,axis=0)
        col_means=np.where(np.isnan(col_means),0.0,col_means)
        nan_mask=np.isnan(raw_data)
        data=np.where(nan_mask,col_means,raw_data)
    
    return data