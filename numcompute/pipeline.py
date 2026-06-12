"""
Minimal Transformer/Estimator protocol:
Preprocessors: fit, transform
Models: fit, predict (no ML model implementation required)
Pipeline chaining
"""

from __future__ import annotations
import numpy as np 
from typing import Any, Callable, Dict, List, Optional, Tuple

class TransformerMixin:
    """
    Mixin that provides a default ``fit_transform`` for any class
    that already implements ``fit`` and ``transform``.
 
    An Estimator sub-class can inherit this mixin if it requires
    fit_transform. 
    """ 
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> np.ndarray:  
              
        """Fit then transform in a single step.
 
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        **fit_params
            Extra keyword arguments forwarded to ``fit``.
 
        Returns
        -------
        X_out : np.ndarray
        """
        return self.fit(X, y, **fit_params).transform(X)


class Transformer(TransformerMixin):
    """
    Base class for all transformers.
 
    Sub-classes must implement ``fit`` and ``transform``.
    ``fit_transform`` is inherited from TransformerMixin which calls both.
    """
 
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "Transformer":
        """Compute statistics needed for the transformation.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : ignored (kept for API compatibility)
        **fit_params : ignored
 
        Returns
        -------
        self
        """
        raise NotImplementedError(f"{type(self).__name__} must implement fit()")
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to X.
 
        Parameters
        ----------
        X : np.ndarray
 
        Returns
        -------
        X_out : np.ndarray
        """
        raise NotImplementedError(f"{type(self).__name__} must implement transform()")

class Estimator:
    """
    Base class for all estimators (models).
 
    Sub-classes must implement ``fit`` and ``predict``.
    """
 
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_params: Any,
    ) -> "Estimator":
        """Fit the model to training data.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        **fit_params : additional keyword arguments
 
        Returns
        -------
        self
        """
        raise NotImplementedError(f"{type(self).__name__} must implement fit()")
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for X.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
 
        Returns
        -------
        y_pred : np.ndarray
        """
        raise NotImplementedError(f"{type(self).__name__} must implement predict()")
 
class Compose(Transformer):
    """
    Wrap a plain callable (function or lambda) as a stateless Transformer.
 
    Because the transform is stateless, ``fit`` is a no-operation function. 
    ``fit_transform`` is inherited from TransformerMixin.
 
    Parameters
    ----------
    func : callable
        ``func(X: np.ndarray) -> np.ndarray``
    name : str, optional
        Label used in repr.
    """
 
    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        name: str = "",
    ) -> None:
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")
        self._func = func
        self._name = name or getattr(func, "__name__", repr(func))
 
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "Compose":
        return self
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the wrapped function to X."""
        result = self._func(X)
        if not isinstance(result, np.ndarray):
            result = np.asarray(result)
        return result
 
    def __repr__(self) -> str:
        return f"Compose(func={self._name})"
    
def _validate_pipeline_steps(steps: List[Tuple[str, Any]]) -> None:
    """Validate pipeline steps at construction time."""
    if not steps:
        raise ValueError("Pipeline requires at least one step.")
 
    names = [name for name, _ in steps]
    if len(names) != len(set(names)):
        raise ValueError("Step names must be unique.")
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Step names must be non-empty strings.")
 
    for name, obj in steps[:-1]:
        if not (hasattr(obj, "fit") and hasattr(obj, "transform")):
            raise TypeError(
                f"Intermediate step '{name}' must implement fit() and transform()."
            )
 
    _, last = steps[-1]
    has_transform = hasattr(last, "fit") and hasattr(last, "transform")
    has_predict   = hasattr(last, "fit") and hasattr(last, "predict")
    if not (has_transform or has_predict):
        raise TypeError(
            "The final step must implement either (fit + transform) "
            "or (fit + predict)."
        )

class Pipeline(TransformerMixin):
    """
    Chain a sequence of transformers with an optional final Estimator.
 
    All intermediate steps must implement ``fit`` and ``transform``.
    The final step may be a Transformer or an Estimator
    (implements ``fit`` + ``predict``).
 
    Parameters
    ----------
    steps : list of (name, object) tuples
        Ordered ``(name, transformer_or_estimator)`` pairs.
        Names must be unique non-empty strings.
 
    Attributes
    ----------
    named_steps : dict
        Mapping of step name → step object for convenient access.
    """
    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        _validate_pipeline_steps(steps)
        self.steps = list(steps)
        self.named_steps: Dict[str, Any] = dict(steps)
 
    def _is_last_estimator(self) -> bool:
        """True when the final step has predict() but no transform()."""
        _, last = self.steps[-1]
        return hasattr(last, "predict") and not hasattr(last, "transform")
    
    def __getitem__(self, name: str) -> Any:
        """Retrieve a step object by name: ``pipe['scale']``."""
        try:
            return self.named_steps[name]
        except KeyError:
            raise KeyError(f"No step named '{name}'.") from None

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "Pipeline":
        """Fit all steps in sequence.
 
        Intermediate steps are fit with ``fit_transform``; the final step
        is fit with ``fit``.  ``y`` and ``**fit_params`` are forwarded to
        every step so supervised transformers work correctly.
 
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        **fit_params : forwarded to every step's fit / fit_transform
 
        Returns
        -------
        self
        """
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y, **fit_params)
 
        _, last = self.steps[-1]
        last.fit(Xt, y, **fit_params)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformer steps to X.

        The final step is included only when it is a Transformer.
        Raises ``RuntimeError`` if the final step is an Estimator — use
        ``predict()`` in that case.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        X_out : np.ndarray
        """
        if self._is_last_estimator():
            raise RuntimeError(
                "The final step is an Estimator; call predict() instead of transform()."
            )
        Xt = X
        for _, step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """Fit then transform through every step.
 
        Only valid when the final step is a Transformer.  Raises
        ``RuntimeError`` if the final step is an Estimator.
 
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        **fit_params : forwarded to every step
 
        Returns
        -------
        X_out : np.ndarray
        """
        if self._is_last_estimator():
            raise RuntimeError(
                "fit_transform() cannot be used when the final step is an "
                "Estimator. Use fit() followed by predict()."
            )
        Xt = X
        for _, step in self.steps:
            Xt = step.fit_transform(Xt, y, **fit_params)
        return Xt 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform through all but the last step, then predict.
 
        Raises ``RuntimeError`` if the final step does not implement
        ``predict()`` (use ``transform()`` or ``fit_transform()`` instead).
 
        Parameters
        ----------
        X : np.ndarray
 
        Returns
        -------
        y_pred : np.ndarray
        """
        _, last = self.steps[-1]
        if not hasattr(last, "predict"):
            raise RuntimeError(
                "The final step does not implement predict(). "
                "Use transform() or fit_transform() instead."
            )
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1][1].predict(Xt)
 
    def get_params(self) -> Dict[str, Any]:
        """Return a dict of step names → step objects."""
        return dict(self.steps)
 
    def __repr__(self) -> str:
        step_str = ", ".join(f"('{n}', {type(s).__name__})" for n, s in self.steps)
        return f"Pipeline([{step_str}])"


class FeatureUnion(Transformer):
    """
    Run multiple transformers in parallel and concatenate their outputs.
 
    Each transformer is provided the same input ``X``.  Outputs are stacked
    column-wise with ``np.hstack``.
 
    Parameters
    ----------
    transformer_list : list of (name, Transformer) tuples
    """
 
    def __init__(self, transformer_list: List[Tuple[str, Any]]) -> None:
        if not transformer_list:
            raise ValueError("FeatureUnion requires at least one transformer.")
        names = [n for n, _ in transformer_list]
        if len(names) != len(set(names)):
            raise ValueError("Transformer names in FeatureUnion must be unique.")
        for name, t in transformer_list:
            if not (hasattr(t, "fit") and hasattr(t, "transform")):
                raise TypeError(
                    f"Transformer '{name}' must implement fit() and transform()."
                )
        self.transformer_list = list(transformer_list)
 
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> "FeatureUnion":
        """Fit each transformer independently on the same X.
 
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        **fit_params : forwarded to every transformer's fit
 
        Returns
        -------
        self

        Note
        ----
         ``**fit_params`` is broadcast to all sub-transformers.  Every
        transformer in the union must accept the same extra keyword
        arguments (or absorb unknown ones via ``**kwargs``).
        """
        for _, t in self.transformer_list:
            t.fit(X, y, **fit_params)
        return self
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X with each transformer and concatenate column-wise.
 
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
 
        Returns
        -------
        X_out : np.ndarray, shape (n_samples, sum_of_output_features)
        """
        parts = []
        for _, t in self.transformer_list:
            out = t.transform(X)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            parts.append(np.ascontiguousarray(out))
        return np.hstack(parts)
 
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """Fit then transform, forwarding y and fit_params to each transformer.
 
        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        **fit_params : forwarded to every transformer
 
        Returns
        -------
        X_out : np.ndarray
        """
        parts = []
        for _, t in self.transformer_list:
            out = t.fit_transform(X, y, **fit_params)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            parts.append(np.ascontiguousarray(out))
        return np.hstack(parts)
 
    def __repr__(self) -> str:
        t_str = ", ".join(
            f"('{n}', {type(t).__name__})" for n, t in self.transformer_list
        )
        return f"FeatureUnion([{t_str}])"