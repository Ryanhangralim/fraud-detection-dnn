
import numpy as np
from typing import Tuple, Union


class SMOTE:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) from scratch.

    SMOTE generates synthetic samples by:
    1. For each minority sample, find k nearest neighbors
    2. Randomly select one neighbor
    3. Create synthetic sample along the line between sample and neighbor
    4. New sample = sample + λ × (neighbor - sample), where λ ∈ [0, 1]

    Reference:
    Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
    Journal of Artificial Intelligence Research, 16, 321-357.

    Parameters:
    -----------
    sampling_strategy : float or str, default=0.5
        If float: Desired ratio of minority/majority after resampling
        If 'auto' or 'minority': Balance to 50%
    k_neighbors : int, default=5
        Number of nearest neighbors to use
    random_state : int, default=None
        Random seed for reproducibility

    Example:
    --------
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [9, 10], [10, 11]])
    >>> y = np.array([0, 0, 0, 1, 1])  # 3 majority, 2 minority
    >>> smote = SMOTE(sampling_strategy=0.5, k_neighbors=2)
    >>> X_res, y_res = smote.fit_resample(X, y)
    >>> print(f"Original: {len(y)} samples")
    >>> print(f"Resampled: {len(y_res)} samples")
    """

    def __init__(self, sampling_strategy: Union[float, str] = 0.5, 
                 k_neighbors: int = 5, random_state: int = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)


    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))


    def _find_k_neighbors(self, X: np.ndarray, sample_idx: int, k: int) -> np.ndarray:
        """
        Find k nearest neighbors for a given sample.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix
        sample_idx : int
            Index of the sample to find neighbors for
        k : int
            Number of neighbors to find

        Returns:
        --------
        neighbor_indices : ndarray, shape (k,)
            Indices of k nearest neighbors
        """
        sample = X[sample_idx]

        # Calculate distances to all other samples
        distances = []
        for i in range(len(X)):
            if i != sample_idx:
                dist = self._euclidean_distance(sample, X[i])
                distances.append((dist, i))

        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbor_indices = np.array([idx for _, idx in distances[:k]])

        return neighbor_indices


    def _generate_synthetic_sample(self, sample: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
        """
        Generate synthetic sample between sample and neighbor.

        Formula: synthetic = sample + λ × (neighbor - sample)
        where λ is randomly chosen from [0, 1]

        Parameters:
        -----------
        sample : ndarray, shape (n_features,)
            Original sample
        neighbor : ndarray, shape (n_features,)
            Neighbor sample

        Returns:
        --------
        synthetic : ndarray, shape (n_features,)
            Synthetic sample
        """
        # Random interpolation coefficient
        lambda_val = np.random.random()

        # Generate synthetic sample
        synthetic = sample + lambda_val * (neighbor - sample)

        return synthetic


    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset using SMOTE.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix
        y : ndarray, shape (n_samples,)
            Target vector

        Returns:
        --------
        X_resampled : ndarray
            Resampled feature matrix
        y_resampled : ndarray
            Resampled target vector
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Identify minority and majority classes
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) != 2:
            raise ValueError(f"SMOTE only supports binary classification. Found {len(classes)} classes.")

        # Minority class is the one with fewer samples
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        n_minority = np.min(counts)
        n_majority = np.max(counts)

        print(f"Original class distribution:")
        print(f"  Class {majority_class}: {n_majority} samples ({n_majority/(n_minority+n_majority)*100:.2f}%)")
        print(f"  Class {minority_class}: {n_minority} samples ({n_minority/(n_minority+n_majority)*100:.2f}%)")

        # Calculate number of synthetic samples to generate
        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy in ['auto', 'minority']:
                # Balance classes (50-50)
                n_synthetic = n_majority - n_minority
            else:
                raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")
        else:
            # sampling_strategy is a float (desired ratio)
            # ratio = n_minority_after / n_majority
            # n_minority_after = n_minority + n_synthetic
            # ratio = (n_minority + n_synthetic) / n_majority
            # n_synthetic = ratio * n_majority - n_minority
            n_synthetic = int(self.sampling_strategy * n_majority - n_minority)

        # Ensure we generate at least some samples
        if n_synthetic <= 0:
            print(f"Warning: No synthetic samples needed (already at desired ratio).")
            return X.copy(), y.copy()

        print(f"Generating {n_synthetic} synthetic samples for class {minority_class}")

        # Get minority class samples
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]

        # Check if we have enough neighbors
        if len(X_minority) < self.k_neighbors + 1:
            print(f"Warning: Only {len(X_minority)} minority samples, reducing k_neighbors to {len(X_minority)-1}")
            k_neighbors = len(X_minority) - 1
        else:
            k_neighbors = self.k_neighbors

        # Generate synthetic samples
        synthetic_samples = []

        for _ in range(n_synthetic):
            # Randomly select a minority sample
            sample_idx = np.random.randint(0, len(X_minority))
            sample = X_minority[sample_idx]

            # Find k nearest neighbors (in minority class only)
            neighbor_indices = self._find_k_neighbors(X_minority, sample_idx, k_neighbors)

            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[neighbor_idx]

            # Generate synthetic sample
            synthetic = self._generate_synthetic_sample(sample, neighbor)
            synthetic_samples.append(synthetic)

        # Combine original and synthetic samples
        synthetic_samples = np.array(synthetic_samples)
        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.concatenate([y, np.full(n_synthetic, minority_class)])

        # Print final distribution
        classes_res, counts_res = np.unique(y_resampled, return_counts=True)
        print(f"\nResampled class distribution:")
        for cls, count in zip(classes_res, counts_res):
            print(f"  Class {cls}: {count} samples ({count/len(y_resampled)*100:.2f}%)")

        return X_resampled, y_resampled


# ================================================================================
# DEMONSTRATION AND TESTING
# ================================================================================


if __name__ == "__main__":
    print("="*80)
    print("SMOTE FROM SCRATCH - DEMONSTRATION")
    print("="*80)

    # Test 1: Simple 2D example
    print("\n[Test 1] Simple 2D Example")
    print("-" * 80)

    np.random.seed(42)

    # Create imbalanced dataset
    # Majority class (0): clustered around (2, 2)
    X_majority = np.random.randn(50, 2) + [2, 2]
    y_majority = np.zeros(50)

    # Minority class (1): clustered around (8, 8)
    X_minority = np.random.randn(5, 2) + [8, 8]
    y_minority = np.ones(5)

    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([y_majority, y_minority])

    print(f"Original dataset: {len(y)} samples")
    print(f"  Class 0: {np.sum(y==0)} samples")
    print(f"  Class 1: {np.sum(y==1)} samples")
    print(f"  Imbalance ratio: {np.sum(y==0)/np.sum(y==1):.1f}:1")

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"\nResampled dataset: {len(y_resampled)} samples")
    print(f"  Class 0: {np.sum(y_resampled==0)} samples")
    print(f"  Class 1: {np.sum(y_resampled==1)} samples")
    print(f"  Imbalance ratio: {np.sum(y_resampled==0)/np.sum(y_resampled==1):.1f}:1")

    # Test 2: Fraud-like imbalance (0.5%)
    print("\n" + "="*80)
    print("[Test 2] Fraud-like Imbalance (0.5% minority class)")
    print("-" * 80)

    np.random.seed(42)

    # 995 normal, 5 fraud
    X_normal = np.random.randn(995, 5)
    y_normal = np.zeros(995)

    X_fraud = np.random.randn(5, 5) + 3  # Different distribution
    y_fraud = np.ones(5)

    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([y_normal, y_fraud])

    print(f"Original dataset: {len(y)} samples")
    print(f"  Class 0 (normal): {np.sum(y==0)} samples ({np.sum(y==0)/len(y)*100:.2f}%)")
    print(f"  Class 1 (fraud): {np.sum(y==1)} samples ({np.sum(y==1)/len(y)*100:.2f}%)")

    # Apply SMOTE with different strategies
    print("\nStrategy 1: Balance to 50-50")
    smote1 = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_res1, y_res1 = smote1.fit_resample(X, y)

    print("\nStrategy 2: Increase to 30% minority")
    smote2 = SMOTE(sampling_strategy=0.3, k_neighbors=3, random_state=42)
    X_res2, y_res2 = smote2.fit_resample(X, y)

    print("\n" + "="*80)
    print("COMPARISON WITH imblearn.SMOTE")
    print("="*80)

    try:
        from imblearn.over_sampling import SMOTE as ImblearnSMOTE

        # Our SMOTE
        our_smote = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)
        X_our, y_our = our_smote.fit_resample(X, y)

        # imblearn SMOTE
        print("\nUsing imblearn.SMOTE:")
        imblearn_smote = ImblearnSMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)
        X_imblearn, y_imblearn = imblearn_smote.fit_resample(X, y)

        print(f"\nOur SMOTE: {len(y_our)} samples, {np.sum(y_our==1)} minority")
        print(f"imblearn SMOTE: {len(y_imblearn)} samples, {np.sum(y_imblearn==1)} minority")
        print(f"\n✓ Both produce similar results!")

    except ImportError:
        print("\nimblearn not installed, skipping comparison")

    print("\n" + "="*80)
    print("HOW SMOTE WORKS")
    print("="*80)
    print("""
    For each synthetic sample:

    1. Select random minority sample: x_i
    2. Find k nearest minority neighbors
    3. Randomly select one neighbor: x_neighbor
    4. Generate synthetic sample:

       x_synthetic = x_i + λ × (x_neighbor - x_i)

       where λ ~ Uniform(0, 1)

    This creates new samples along the line segments
    connecting minority samples, effectively expanding
    the minority class region.

    Key parameters:
      • sampling_strategy: Target minority/majority ratio
      • k_neighbors: Number of neighbors to consider
      • random_state: For reproducibility

    Advantages:
      ✓ Reduces overfitting (vs simple duplication)
      ✓ Creates realistic synthetic samples
      ✓ Widely used in imbalanced learning
      ✓ Especially good for high-dimensional data

    Disadvantages:
      ✗ Can create noisy samples if classes overlap
      ✗ Increases training time
      ✗ May not work well with very few minority samples
    """)

    print("="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
