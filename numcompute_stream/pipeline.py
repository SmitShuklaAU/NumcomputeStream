"""
pipeline.py
-----------
Streaming pipeline operations for numcompute_stream.

Extends the Assignment 2.1 static Pipeline and FeatureUnion to support
incremental chunk processing via `.partial_fit()`. 
"""

from __future__ import annotations
import numpy as np
from typing import Any, Optional

# Import base components from Assignment 2.1
from numcompute.pipeline import (
    Pipeline,
    FeatureUnion,
    Transformer,
    Estimator,
    Compose,
    TransformerMixin
)

__all__ = [
    "StreamPipeline",
    "StreamFeatureUnion",
    "Transformer",
    "Estimator",
    "Compose",
    "TransformerMixin"
]

class StreamPipeline(Pipeline):
    """
    Streaming version of the Pipeline.
    
    Chains multiple streaming transformers and an optional final streaming Estimator.
    Supports `.partial_fit()` to incrementally train all steps on incoming chunks.
    """
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "StreamPipeline":
        """
        Incrementally fit all steps in sequence with a new chunk of data.
        
        For each intermediate transformer, it calls `.partial_fit()` to update
        its state, then `.transform()` to process the chunk for the next step.
        Finally, it calls `.partial_fit()` on the last step.
        
        Parameters
        ----------
        X : np.ndarray
            Incoming chunk of features.
        y : np.ndarray or None
            Incoming chunk of targets.
        **fit_params : additional keyword arguments forwarded to every step.
        
        Returns
        -------
        self
        """
        Xt = X
        # Route through all transformers
        for name, step in self.steps[:-1]:
            if not hasattr(step, "partial_fit"):
                raise AttributeError(f"Intermediate step '{name}' does not implement partial_fit().")
            
            # Update the transformer's state, then transform the chunk for the next step
            step.partial_fit(Xt, y, **fit_params)
            Xt = step.transform(Xt)
            
        # Route to the final step (Transformer or Estimator)
        last_name, last_step = self.steps[-1]
        if not hasattr(last_step, "partial_fit"):
            raise AttributeError(f"Final step '{last_name}' does not implement partial_fit().")
            
        last_step.partial_fit(Xt, y, **fit_params)
        return self


class StreamFeatureUnion(FeatureUnion):
    """
    Streaming version of FeatureUnion.
    
    Runs multiple streaming transformers in parallel on the same chunk of data,
    concatenating their outputs. 
    """
    
    def partial_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "StreamFeatureUnion":
        """
        Incrementally fit each transformer independently on the same chunk X.
        
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        **fit_params : forwarded to every transformer's partial_fit
        
        Returns
        -------
        self
        """
        for name, step in self.transformer_list:
            if not hasattr(step, "partial_fit"):
                raise AttributeError(f"Transformer '{name}' does not implement partial_fit().")
            
            step.partial_fit(X, y, **fit_params)
            
        return self