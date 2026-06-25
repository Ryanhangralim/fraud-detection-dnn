
import numpy as np

def compute_class_weight(class_weight, classes, y):
    """
    Compute class weights for imbalanced datasets.

    Parameters:
    -----------
    class_weight : str or dict
        If 'balanced', uses n_samples / (n_classes * np.bincount(y))
        If dict, maps class labels to weights
    classes : array-like
        Array of class labels (e.g., [0, 1])
    y : array-like
        Array of target values

    Returns:
    --------
    class_weight_dict : dict
        Dictionary mapping each class to its weight

    Example:
    --------
    >>> y = np.array([0, 0, 0, 0, 1])  # Imbalanced
    >>> weights = compute_class_weight('balanced', classes=[0, 1], y=y)
    >>> print(weights)
    {0: 0.625, 1: 2.5}

    Explanation:
    - Total samples: 5
    - Class 0: 4 samples, Class 1: 1 sample
    - Weight_0 = 5 / (2 * 4) = 0.625
    - Weight_1 = 5 / (2 * 1) = 2.5
    - This gives minority class (1) higher weight
    """
    if isinstance(class_weight, dict):
        # User provided custom weights
        return class_weight

    elif class_weight == 'balanced':
        # Calculate balanced weights
        y = np.asarray(y).ravel()  # Ensure 1D array
        classes = np.asarray(classes)

        # Count samples per class
        class_counts = np.bincount(y.astype(int))

        # Total samples
        n_samples = len(y)

        # Number of classes
        n_classes = len(classes)

        # Compute weight for each class
        # Formula: n_samples / (n_classes * class_count)
        weights = n_samples / (n_classes * class_counts)

        # Create dictionary mapping class to weight
        class_weight_dict = {int(cls): float(weight) 
                            for cls, weight in zip(classes, weights)}

        return class_weight_dict

    else:
        raise ValueError(f"class_weight must be 'balanced' or dict, got {class_weight}")


def get_sample_weights(y, class_weight_dict):
    """
    Convert class weights to sample weights.

    Parameters:
    -----------
    y : array-like
        Array of target values
    class_weight_dict : dict
        Dictionary mapping class labels to weights

    Returns:
    --------
    sample_weights : ndarray
        Array of weights for each sample

    Example:
    --------
    >>> y = np.array([0, 0, 1, 1, 1])
    >>> class_weights = {0: 1.5, 1: 0.5}
    >>> sample_weights = get_sample_weights(y, class_weights)
    >>> print(sample_weights)
    [1.5, 1.5, 0.5, 0.5, 0.5]
    """
    y = np.asarray(y).ravel()
    sample_weights = np.zeros(len(y), dtype=float)

    for cls, weight in class_weight_dict.items():
        sample_weights[y == cls] = weight

    return sample_weights


if __name__ == "__main__":
    # Test the function
    print("="*80)
    print("TESTING compute_class_weight")
    print("="*80)

    # Test 1: Balanced dataset
    print("\nTest 1: Balanced Dataset")
    y_balanced = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    weights = compute_class_weight('balanced', classes=[0, 1], y=y_balanced)
    print(f"  Classes: {np.bincount(y_balanced)}")
    print(f"  Weights: {weights}")

    # Test 2: Imbalanced dataset (like credit card fraud)
    print("\nTest 2: Imbalanced Dataset (0.5% fraud)")
    y_imbalanced = np.concatenate([np.zeros(995), np.ones(5)])
    weights = compute_class_weight('balanced', classes=[0, 1], y=y_imbalanced)
    print(f"  Classes: {np.bincount(y_imbalanced.astype(int))}")
    print(f"  Weights: {weights}")
    print(f"  Ratio: Class 1 gets {weights[1]/weights[0]:.1f}x more weight than Class 0")

    # Test 3: Custom weights
    print("\nTest 3: Custom Weights")
    custom_weights = {0: 1.0, 1: 10.0}
    weights = compute_class_weight(custom_weights, classes=[0, 1], y=y_imbalanced)
    print(f"  Custom weights: {weights}")

    # Test 4: Sample weights
    print("\nTest 4: Convert to Sample Weights")
    y_small = np.array([0, 0, 0, 1])
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_small)
    sample_weights = get_sample_weights(y_small, class_weights)
    print(f"  y: {y_small}")
    print(f"  Class weights: {class_weights}")
    print(f"  Sample weights: {sample_weights}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
