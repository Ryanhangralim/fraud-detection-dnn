
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional

def train_test_split(*arrays, test_size: float = 0.25, random_state: Optional[int] = None, 
                     stratify: Optional[np.ndarray] = None, shuffle: bool = True):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    -----------
    *arrays : sequence of indexables
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split (0.0 to 1.0).
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output.
    stratify : array-like, optional
        If not None, data is split in a stratified fashion, using this as the class labels.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    Returns:
    --------
    splitting : list
        List containing train-test split of inputs.
        [X_train, X_test, y_train, y_test] if 2 arrays passed.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Get number of samples from first array
    n_samples = len(arrays[0])

    # Validate all arrays have same length
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All input arrays must have the same number of samples")

    # Calculate split index
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size should be between 0.0 and 1.0, got {test_size}")

    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # Handle stratified split
    if stratify is not None:
        # Convert to numpy array
        if isinstance(stratify, (pd.Series, pd.DataFrame)):
            stratify = stratify.values
        if stratify.ndim > 1:
            stratify = stratify.ravel()

        # Get unique classes and their counts
        classes, class_counts = np.unique(stratify, return_counts=True)

        # Check if stratification is possible
        for cls, count in zip(classes, class_counts):
            n_test_class = int(count * test_size)
            if n_test_class < 1:
                raise ValueError(f"The least populated class has only {count} members, "
                               f"which is too few for test_size={test_size}")

        # Create stratified indices
        train_indices = []
        test_indices = []

        for cls in classes:
            # Get indices for this class
            cls_indices = np.where(stratify == cls)[0]

            # Shuffle if needed
            if shuffle:
                np.random.shuffle(cls_indices)

            # Split proportionally
            n_test_cls = int(len(cls_indices) * test_size)

            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])

        # Convert to numpy arrays
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Shuffle the combined indices to mix classes
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

    else:
        # Non-stratified split
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    # Split all arrays
    result = []
    for arr in arrays:
        # Convert to appropriate format
        is_dataframe = isinstance(arr, pd.DataFrame)
        is_series = isinstance(arr, pd.Series)

        if is_dataframe or is_series:
            train_split = arr.iloc[train_indices]
            test_split = arr.iloc[test_indices]

            # Reset index
            train_split = train_split.reset_index(drop=True)
            test_split = test_split.reset_index(drop=True)
        else:
            # Numpy array or list
            if isinstance(arr, list):
                arr = np.array(arr)

            train_split = arr[train_indices]
            test_split = arr[test_indices]

        result.extend([train_split, test_split])

    return result


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRAIN_TEST_SPLIT FROM SCRATCH - DEMONSTRATION")
    print("="*70)

    # Test 1: Basic split
    print("\n[Test 1] Basic split without stratification")
    print("-" * 70)

    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Original: X={X.shape}, y={y.shape}")
    print(f"Train: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Test 2: Stratified split
    print("\n[Test 2] Stratified split (imbalanced classes)")
    print("-" * 70)

    # Create imbalanced dataset
    X = np.arange(1000).reshape(1000, 1)
    y = np.array([0] * 970 + [1] * 30)  # 3% positive class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Original class distribution: {np.bincount(y)} ({y.mean()*100:.2f}% positive)")
    print(f"Train class distribution: {np.bincount(y_train)} ({y_train.mean()*100:.2f}% positive)")
    print(f"Test class distribution: {np.bincount(y_test)} ({y_test.mean()*100:.2f}% positive)")

    # Test 3: Pandas DataFrame
    print("\n[Test 3] With pandas DataFrame")
    print("-" * 70)

    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'label': np.random.randint(0, 2, 100)
    })

    X_df = df[['feature1', 'feature2']]
    y_df = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.3, random_state=42, stratify=y_df
    )

    print(f"Original: X={X_df.shape}, y={y_df.shape}")
    print(f"Train: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"X_train type: {type(X_train)}")
    print(f"y_train type: {type(y_train)}")

    # Test 4: Comparison with sklearn
    print("\n[Test 4] Comparison with sklearn")
    print("-" * 70)

    try:
        from sklearn.model_selection import train_test_split as sklearn_split

        X = np.arange(100).reshape(100, 1)
        y = np.array([0] * 80 + [1] * 20)

        # Custom split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Sklearn split
        X_train2, X_test2, y_train2, y_test2 = sklearn_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Custom implementation:")
        print(f"  Train: {X_train1.shape}, class dist: {np.bincount(y_train1)}")
        print(f"  Test: {X_test1.shape}, class dist: {np.bincount(y_test1)}")

        print(f"\nSklearn implementation:")
        print(f"  Train: {X_train2.shape}, class dist: {np.bincount(y_train2)}")
        print(f"  Test: {X_test2.shape}, class dist: {np.bincount(y_test2)}")

        print(f"\nClass distribution match: Train={np.array_equal(np.bincount(y_train1), np.bincount(y_train2))}, "
              f"Test={np.array_equal(np.bincount(y_test1), np.bincount(y_test2))}")

    except ImportError:
        print("sklearn not available, skipping comparison")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
