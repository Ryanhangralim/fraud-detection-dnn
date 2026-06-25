
import numpy as np
import pandas as pd
from typing import Union, Optional

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    labels : array-like, optional
        List of labels to index the matrix. If None, labels are inferred.

    Returns:
    --------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates
        the number of samples with true label i and predicted label j.

    For binary classification (2 classes):
        [[TN, FP],
         [FN, TP]]
    """
    # Convert to numpy arrays
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Check same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Get unique labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    n_labels = len(labels)

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)

    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1

    return cm


# ============================================================================
# BASIC METRICS
# ============================================================================

def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, zero_division=0):
    """
    Compute the precision.

    Precision = TP / (TP + FP)
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        if (tp + fp) > 0:
            return tp / (tp + fp)
        else:
            return zero_division
    else:
        raise ValueError("Only binary classification is supported")


def recall_score(y_true, y_pred, zero_division=0):
    """
    Compute the recall (sensitivity).

    Recall = TP / (TP + FN)
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) > 0:
            return tp / (tp + fn)
        else:
            return zero_division
    else:
        raise ValueError("Only binary classification is supported")


def f1_score(y_true, y_pred, average='binary', zero_division=0):
    """
    Compute the F1 score, also known as balanced F-score or F-measure.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    # Convert to numpy arrays
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.values

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division

        # Calculate F1
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = zero_division

        return f1
    else:
        raise ValueError("Only binary classification is supported")


# ============================================================================
# ROC AUC SCORE
# ============================================================================

def roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Uses the trapezoidal rule to compute the area under the ROC curve.
    """
    # Convert to numpy arrays
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.values

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Check valid input
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    # Sort by predicted score (descending)
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Get unique thresholds
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.concatenate([distinct_value_indices, [len(y_true_sorted) - 1]])

    # Calculate TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = 1 + threshold_indices - tps

    # Calculate total positives and negatives
    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        raise ValueError("ROC AUC score is not defined when only one class is present")

    # Calculate TPR and FPR
    tpr = tps / total_pos
    fpr = fps / total_neg

    # Add (0, 0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Calculate AUC using trapezoidal rule
    auc = np.trapezoid(tpr, fpr)

    return auc


# ============================================================================
# AVERAGE PRECISION SCORE (PR AUC)
# ============================================================================

def average_precision_score(y_true, y_score):
    """
    Compute average precision (AP) from prediction scores.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight.
    """
    # Convert to numpy arrays
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.values

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Check valid input
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    # Sort by predicted score (descending)
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Calculate precision and recall at each position
    tp_sum = np.cumsum(y_true_sorted)

    # Calculate precision at each threshold
    precisions = tp_sum / np.arange(1, len(y_true_sorted) + 1)

    # Calculate recall at each threshold
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0

    recalls = tp_sum / total_positives

    # Calculate average precision using trapezoidal approximation
    # Add sentinel values at the beginning and end
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find where recall changes
    recall_changes = np.where(np.diff(recalls))[0]

    # Calculate average precision as area under curve
    ap = np.sum((recalls[recall_changes + 1] - recalls[recall_changes]) * 
                precisions[recall_changes + 1])

    return ap


# ============================================================================
# AVERAGE RECALL SCORE (NEW!)
# ============================================================================

def average_recall_score(y_true, y_score, thresholds=None):
    """
    Compute average recall across multiple thresholds.

    This metric is useful for evaluating model performance across different
    decision thresholds, particularly in imbalanced classification problems.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_score : array-like
        Target scores (probability estimates of the positive class).
    thresholds : array-like, optional
        Thresholds to evaluate. If None, uses [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
    --------
    avg_recall : float
        Average recall across all thresholds.
    recalls : list of float
        Recall at each threshold.
    """
    # Convert to numpy arrays
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.values

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Default thresholds
    if thresholds is None:
        thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    else:
        thresholds = np.asarray(thresholds)

    # Calculate recall at each threshold
    recalls = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        recalls.append(recall)

    # Calculate average recall
    avg_recall = np.mean(recalls)

    return avg_recall, recalls


# ============================================================================
# BALANCED ACCURACY SCORE (NEW!)
# ============================================================================

def balanced_accuracy_score(y_true, y_pred):
    """
    Compute the balanced accuracy.

    The balanced accuracy is the average of recall obtained on each class.
    It's useful for imbalanced datasets.

    Balanced Accuracy = (Sensitivity + Specificity) / 2
                      = (Recall_positive + Recall_negative) / 2

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns:
    --------
    balanced_acc : float
        Balanced accuracy score.
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        # Sensitivity (recall for positive class)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity (recall for negative class)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Balanced accuracy
        balanced_acc = (sensitivity + specificity) / 2

        return balanced_acc
    else:
        raise ValueError("Only binary classification is supported")


# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

def classification_report(y_true, y_pred, target_names=None, digits=4):
    """
    Build a text report showing the main classification metrics.
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        raise ValueError("Only binary classification is supported")

    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics for each class
    # Class 0 (negative)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    support_0 = tn + fp

    # Class 1 (positive)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    support_1 = tp + fn

    # Overall metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    # Weighted metrics
    total = support_0 + support_1
    weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / total
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / total
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total

    # Default target names
    if target_names is None:
        target_names = ['0', '1']

    # Build report
    headers = ['precision', 'recall', 'f1-score', 'support']
    head_fmt = '{:>15}' * (len(headers) + 1)
    report_fmt = '{:>15}' + '{:>15.{digits}f}' * 3 + '{:>15}'

    lines = []
    lines.append(head_fmt.format('', *headers))
    lines.append('')

    # Class 0
    lines.append(report_fmt.format(
        target_names[0], precision_0, recall_0, f1_0, support_0, digits=digits
    ))

    # Class 1
    lines.append(report_fmt.format(
        target_names[1], precision_1, recall_1, f1_1, support_1, digits=digits
    ))

    lines.append('')

    # Accuracy
    lines.append('{:>15}{:>15}{:>15}{:>15.{digits}f}{:>15}'.format(
        'accuracy', '', '', accuracy, total, digits=digits
    ))

    # Macro avg
    lines.append(report_fmt.format(
        'macro avg', macro_precision, macro_recall, macro_f1, total, digits=digits
    ))

    # Weighted avg
    lines.append(report_fmt.format(
        'weighted avg', weighted_precision, weighted_recall, weighted_f1, total, digits=digits
    ))

    return '\n'.join(lines)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CLASSIFICATION METRICS FROM SCRATCH - DEMONSTRATION")
    print("="*80)

    # Test data
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    y_score = np.random.rand(20)  # Predicted probabilities

    print("\n[Test 1] Basic Metrics")
    print("-" * 80)

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Confusion Matrix: [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    print("\n[Test 2] Threshold-Independent Metrics")
    print("-" * 80)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print("\n[Test 3] NEW: Average Recall Score")
    print("-" * 80)

    avg_recall, recalls = average_recall_score(y_true, y_score)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"Recall at different thresholds:")
    for t, r in zip(thresholds, recalls):
        print(f"  Threshold {t:.1f}: {r:.4f}")
    print(f"\nAverage Recall: {avg_recall:.4f}")

    print("\n[Test 4] NEW: Balanced Accuracy Score")
    print("-" * 80)

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Regular Accuracy:  {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  (Better for imbalanced datasets)")

    print("\n[Test 5] Classification Report")
    print("-" * 80)

    report = classification_report(y_true, y_pred, target_names=['Non-Fraud', 'Fraud'])
    print(report)

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80)
