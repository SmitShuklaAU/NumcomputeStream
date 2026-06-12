import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
        y_true (np.ndarray): Array of true target values. Shape (N,).
        y_pred (np.ndarray): Array of predicted target values. Shape (N,).

    Returns:
        float: The mean squared error.

    Raises:
        ValueError: If input shapes do not match.

    Complexity:
        Time: O(N)
        Space: O(N) for intermediate differences.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
    
    return float(np.nanmean((y_true - y_pred) ** 2))

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the classification accuracy.

    Parameters:
        y_true (np.ndarray): True class labels. Shape (N,).
        y_pred (np.ndarray): Predicted class labels. Shape (N,).

    Returns:
        float: Accuracy score between 0.0 and 1.0.
    
    Complexity:
        Time: O(N)
        Space: O(N) for boolean array.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
        
    return float(np.mean(y_true == y_pred))

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> tuple:
    """
    Computes precision, recall, and F1-score for binary classification.
    Uses numerical stability (epsilon) to prevent division by zero.

    Parameters:
        y_true (np.ndarray): True class labels. Shape (N,).
        y_pred (np.ndarray): Predicted class labels. Shape (N,).
        pos_label (int): The label considered as the positive class.

    Returns:
        tuple: (precision, recall, f1_score)
    
    Complexity:
        Time: O(N)
        Space: O(N) for boolean masks.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")

    true_pos = np.sum((y_pred == pos_label) & (y_true == pos_label))
    false_pos = np.sum((y_pred == pos_label) & (y_true != pos_label))
    false_neg = np.sum((y_pred != pos_label) & (y_true == pos_label))

    eps = np.finfo(float).eps  # Numerical stability
    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return float(precision), float(recall), float(f1)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Computes the confusion matrix using highly optimized vectorized bincount.

    Parameters:
        y_true (np.ndarray): True integer class labels [0, num_classes-1]. Shape (N,).
        y_pred (np.ndarray): Predicted integer class labels [0, num_classes-1]. Shape (N,).
        num_classes (int): Total number of distinct classes.

    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes).
                    Rows represent actual/true labels, columns predicted labels.

    Complexity:
        Time: O(N)
        Space: O(N) for linear indices, O(num_classes^2) for the output matrix.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between true and predicted labels.")
    
    # Vectorized 1D index mapping: true_label * num_classes + pred_label
    linear_indices = y_true * num_classes + y_pred
    bincount = np.bincount(linear_indices, minlength=num_classes**2)
    return bincount.reshape((num_classes, num_classes))

def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Handles ties in probability scores.

    Parameters:
        y_true (np.ndarray): True binary labels {0, 1}. Shape (N,).
        y_scores (np.ndarray): Target scores/probabilities. Shape (N,).

    Returns:
        float: AUC score.
    
    Complexity:
        Time: O(N log N) due to sorting.
        Space: O(N) for sorted arrays and cumsums.
    """
    if y_true.shape != y_scores.shape:
        raise ValueError("Shape mismatch between labels and scores.")

    # Sort scores descending
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    # Find distinct threshold indices to handle ties properly
    distinct_value_indices = np.where(np.diff(y_scores_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    # Calculate cumulative true positives and false positives
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add (0,0) to the ROC curve origin
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    # Normalize to get rates
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    # Integrate using the trapezoidal rule
    return float(np.trapezoid(y=tpr, x=fpr))