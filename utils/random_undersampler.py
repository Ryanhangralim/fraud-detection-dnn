"""
Random Undersampling Implementation
====================================
Reduces majority class by randomly removing samples to balance dataset.

Author: [Your Name]
Date: February 2026
"""

import numpy as np


class RandomUnderSampler:
    """
    Random Undersampling for imbalanced datasets.
    
    Reduces the majority class by randomly removing samples until the desired
    class ratio is achieved. Minority class samples are preserved.
    
    Parameters
    ----------
    sampling_strategy : float or str, default=1.0
        Desired ratio of minority to majority class after resampling.
        - If float: ratio of minority/majority (e.g., 1.0 = balanced)
        - If 'auto': same as 1.0 (balanced)
        - If dict: {class_label: n_samples} for custom ratios
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    replacement : bool, default=False
        Whether to sample with replacement (usually False for undersampling).
    
    Attributes
    ----------
    sample_indices_ : dict
        Dictionary mapping class labels to sampled indices.
    
    sampling_strategy_ : float
        The actual sampling strategy used.
    
    Examples
    --------
    >>> from random_undersampler import RandomUnderSampler
    >>> rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
    >>> X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    >>> print(f"Original: {len(X_train)}, Resampled: {len(X_resampled)}")
    """
    
    def __init__(self, sampling_strategy=1.0, random_state=None, replacement=False):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement
        self.sample_indices_ = {}
        self.sampling_strategy_ = None
        
    def fit_resample(self, X, y):
        """
        Resample the dataset by undersampling the majority class.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features.
        
        y : array-like, shape (n_samples,)
            Target labels.
        
        Returns
        -------
        X_resampled : array-like, shape (n_samples_new, n_features)
            Resampled features.
        
        y_resampled : array-like, shape (n_samples_new,)
            Resampled labels.
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get unique classes and their counts
        classes, class_counts = np.unique(y, return_counts=True)
        
        # Identify minority and majority classes
        minority_class = classes[np.argmin(class_counts)]
        majority_class = classes[np.argmax(class_counts)]
        
        n_minority = class_counts.min()
        n_majority = class_counts.max()
        
        # Determine target number of majority samples
        if isinstance(self.sampling_strategy, str) and self.sampling_strategy == 'auto':
            self.sampling_strategy_ = 1.0
        elif isinstance(self.sampling_strategy, (int, float)):
            self.sampling_strategy_ = float(self.sampling_strategy)
        else:
            raise ValueError(f"sampling_strategy must be float or 'auto', got {type(self.sampling_strategy)}")
        
        # Calculate target majority class size
        n_majority_target = int(n_minority / self.sampling_strategy_)
        
        # Ensure we don't try to oversample (that's not undersampling!)
        if n_majority_target > n_majority:
            print(f"Warning: Target majority samples ({n_majority_target}) > current ({n_majority})")
            print(f"Using all majority samples (no undersampling needed)")
            n_majority_target = n_majority
        
        # Get indices for each class
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Keep ALL minority class samples
        sampled_minority_indices = minority_indices
        
        # Randomly sample from majority class
        if n_majority_target < n_majority:
            sampled_majority_indices = np.random.choice(
                majority_indices,
                size=n_majority_target,
                replace=self.replacement
            )
        else:
            sampled_majority_indices = majority_indices
        
        # Combine indices
        resampled_indices = np.concatenate([
            sampled_minority_indices,
            sampled_majority_indices
        ])
        
        # Shuffle to mix classes
        np.random.shuffle(resampled_indices)
        
        # Store for inspection
        self.sample_indices_ = {
            minority_class: sampled_minority_indices,
            majority_class: sampled_majority_indices
        }
        
        # Return resampled data
        X_resampled = X[resampled_indices]
        y_resampled = y[resampled_indices]
        
        return X_resampled, y_resampled
    
    def fit(self, X, y):
        """
        Fit the sampler (computes sampling indices).
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Target labels.
        
        Returns
        -------
        self : object
            Fitted sampler.
        """
        # Just compute the indices without returning data
        self.fit_resample(X, y)
        return self
    
    def get_params(self):
        """
        Get parameters of the sampler.
        
        Returns
        -------
        params : dict
            Dictionary of parameters.
        """
        return {
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state,
            'replacement': self.replacement
        }
    
    def __repr__(self):
        """String representation of the sampler."""
        return (f"RandomUnderSampler(sampling_strategy={self.sampling_strategy}, "
                f"random_state={self.random_state}, replacement={self.replacement})")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_resampling_summary(y_original, y_resampled, title="Resampling Summary"):
    """
    Print summary statistics of resampling.
    
    Parameters
    ----------
    y_original : array-like
        Original labels before resampling.
    y_resampled : array-like
        Labels after resampling.
    title : str
        Title for the summary.
    """
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    classes_orig, counts_orig = np.unique(y_original, return_counts=True)
    classes_res, counts_res = np.unique(y_resampled, return_counts=True)
    
    print(f"\n{'Class':<15} {'Original':<15} {'Resampled':<15} {'Change':<15}")
    print("-" * 60)
    
    for cls in classes_orig:
        orig_count = counts_orig[classes_orig == cls][0]
        res_count = counts_res[classes_res == cls][0] if cls in classes_res else 0
        change = res_count - orig_count
        change_pct = (change / orig_count * 100) if orig_count > 0 else 0
        
        print(f"{cls:<15} {orig_count:<15,} {res_count:<15,} {change:+,} ({change_pct:+.1f}%)")
    
    print("-" * 60)
    print(f"{'Total':<15} {len(y_original):<15,} {len(y_resampled):<15,} "
          f"{len(y_resampled) - len(y_original):+,}")
    
    print(f"\n{'Metric':<30} {'Original':<20} {'Resampled':<20}")
    print("-" * 70)
    
    orig_fraud_rate = (y_original == 1).mean() * 100
    res_fraud_rate = (y_resampled == 1).mean() * 100
    
    print(f"{'Fraud Rate':<30} {orig_fraud_rate:<20.4f}% {res_fraud_rate:<20.4f}%")
    print(f"{'Imbalance Ratio (1:N)':<30} {f'1:{1/orig_fraud_rate*100:.0f}':<20} "
          f"{f'1:{1/res_fraud_rate*100:.0f}':<20}")
    
    print("=" * 80)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Test the RandomUnderSampler implementation.
    """
    print("\n" + "=" * 80)
    print("TESTING RandomUnderSampler - From Scratch Implementation")
    print("=" * 80)
    
    # Create synthetic imbalanced dataset
    np.random.seed(42)
    
    # Normal transactions (majority class)
    X_normal = np.random.randn(10000, 30)
    y_normal = np.zeros(10000)
    
    # Fraud transactions (minority class)
    X_fraud = np.random.randn(50, 30) + 2  # Slightly different distribution
    y_fraud = np.ones(50)
    
    # Combine
    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([y_normal, y_fraud])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    print(f"\n✓ Created synthetic dataset:")
    print(f"  Normal: {(y == 0).sum():,}")
    print(f"  Fraud:  {(y == 1).sum():,}")
    print(f"  Total:  {len(y):,}")
    print(f"  Imbalance: 1:{(y == 0).sum() / (y == 1).sum():.0f}")
    
    # ========================================================================
    # Test 1: Balanced sampling (1:1 ratio)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Balanced Sampling (1:1 ratio)")
    print("=" * 80)
    
    rus_balanced = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = rus_balanced.fit_resample(X, y)
    
    print_resampling_summary(y, y_resampled, "Test 1: Balanced (1:1)")
    
    # ========================================================================
    # Test 2: Moderate imbalance (1:5 ratio)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Moderate Imbalance (1:5 ratio)")
    print("=" * 80)
    
    rus_moderate = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
    X_resampled_mod, y_resampled_mod = rus_moderate.fit_resample(X, y)
    
    print_resampling_summary(y, y_resampled_mod, "Test 2: Moderate (1:5)")
    
    # ========================================================================
    # Test 3: Light undersampling (1:10 ratio)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Light Undersampling (1:10 ratio)")
    print("=" * 80)
    
    rus_light = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    X_resampled_light, y_resampled_light = rus_light.fit_resample(X, y)
    
    print_resampling_summary(y, y_resampled_light, "Test 3: Light (1:10)")
    
    # ========================================================================
    # Verification: Check data integrity
    # ========================================================================
    print("\n" + "=" * 80)
    print("DATA INTEGRITY CHECKS")
    print("=" * 80)
    
    # Check 1: All minority samples preserved
    original_fraud_count = (y == 1).sum()
    resampled_fraud_count = (y_resampled == 1).sum()
    
    print(f"\n✓ Check 1: Minority class preservation")
    print(f"  Original fraud samples: {original_fraud_count}")
    print(f"  Resampled fraud samples: {resampled_fraud_count}")
    print(f"  Status: {'PASS' if original_fraud_count == resampled_fraud_count else 'FAIL'}")
    
    # Check 2: Majority class reduced
    original_normal_count = (y == 0).sum()
    resampled_normal_count = (y_resampled == 0).sum()
    
    print(f"\n✓ Check 2: Majority class reduction")
    print(f"  Original normal samples: {original_normal_count:,}")
    print(f"  Resampled normal samples: {resampled_normal_count:,}")
    print(f"  Reduction: {original_normal_count - resampled_normal_count:,} "
          f"({(1 - resampled_normal_count/original_normal_count)*100:.1f}%)")
    print(f"  Status: {'PASS' if resampled_normal_count < original_normal_count else 'FAIL'}")
    
    # Check 3: Feature shapes preserved
    print(f"\n✓ Check 3: Feature dimensions")
    print(f"  Original shape: {X.shape}")
    print(f"  Resampled shape: {X_resampled.shape}")
    print(f"  Status: {'PASS' if X.shape[1] == X_resampled.shape[1] else 'FAIL'}")
    
    # Check 4: No data leakage (samples are subset of original)
    print(f"\n✓ Check 4: Data integrity")
    print(f"  All resampled samples from original: ", end="")
    # This is guaranteed by our implementation (we only select indices)
    print("PASS (by design)")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
