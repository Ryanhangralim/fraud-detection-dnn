
import numpy as np
import pandas as pd

class StandardScaler:
    """
    StandardScaler implementation from scratch.

    Standardizes features by removing the mean and scaling to unit variance.
    Formula: z = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
        self.is_fitted_ = False

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns:
        --------
        self : object
            Fitted scaler.
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)

        # Calculate mean and std for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)  # ddof=0 for population std (same as sklearn)

        # Handle zero std (constant features)
        # Sklearn uses std=1 for constant features to avoid division by zero
        self.std_[self.std_ == 0] = 1.0

        self.n_features_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.

        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        if not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")

        # Convert to numpy array if needed
        original_type = type(X)
        is_dataframe = isinstance(X, pd.DataFrame)
        columns = None
        index = None

        if is_dataframe:
            columns = X.columns
            index = X.index
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)

        # Check feature dimension
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but scaler was fitted with {self.n_features_} features")

        # Apply standardization: (X - mean) / std
        X_scaled = (X - self.mean_) / self.std_

        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            X_scaled = pd.DataFrame(X_scaled, columns=columns, index=index)

        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data to fit and transform.

        Returns:
        --------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The data to inverse transform.

        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Original data.
        """
        if not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")

        # Convert to numpy array if needed
        is_dataframe = isinstance(X, pd.DataFrame)
        columns = None
        index = None

        if is_dataframe:
            columns = X.columns
            index = X.index
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)

        # Apply inverse transformation: X_original = (X_scaled * std) + mean
        X_original = (X * self.std_) + self.mean_

        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            X_original = pd.DataFrame(X_original, columns=columns, index=index)

        return X_original

    def get_params(self):
        """
        Get parameters of the scaler.

        Returns:
        --------
        params : dict
            Dictionary containing mean and std.
        """
        if not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")

        return {
            'mean': self.mean_,
            'std': self.std_,
            'n_features': self.n_features_
        }

    def __repr__(self):
        if self.is_fitted_:
            return f"StandardScaler(n_features={self.n_features_}, fitted=True)"
        else:
            return "StandardScaler(fitted=False)"


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STANDARDSCALER FROM SCRATCH - DEMONSTRATION")
    print("="*70)

    # Create sample data
    print("\n[Test 1] Basic functionality with numpy array")
    print("-" * 70)

    X_train = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ])

    X_test = np.array([
        [11, 12],
        [13, 14]
    ])

    print(f"Training data:\n{X_train}")
    print(f"\nTest data:\n{X_test}")

    # Fit and transform
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFitted parameters:")
    print(f"  Mean: {scaler.mean_}")
    print(f"  Std:  {scaler.std_}")

    print(f"\nScaled training data:\n{X_train_scaled}")
    print(f"\nScaled test data:\n{X_test_scaled}")

    # Verify: scaled data should have mean ≈ 0 and std ≈ 1
    print(f"\nVerification (training data):")
    print(f"  Mean: {np.mean(X_train_scaled, axis=0)}")
    print(f"  Std:  {np.std(X_train_scaled, axis=0)}")

    # Inverse transform
    X_train_original = scaler.inverse_transform(X_train_scaled)
    print(f"\nInverse transformed (should match original):\n{X_train_original}")

    # Test with pandas DataFrame
    print("\n" + "="*70)
    print("[Test 2] With pandas DataFrame")
    print("-" * 70)

    df_train = pd.DataFrame({
        'Time': [100, 200, 300, 400, 500],
        'Amount': [10.5, 25.3, 50.0, 100.2, 200.8]
    })

    df_test = pd.DataFrame({
        'Time': [600, 700],
        'Amount': [150.0, 300.5]
    })

    print(f"Training DataFrame:\n{df_train}")
    print(f"\nTest DataFrame:\n{df_test}")

    scaler2 = StandardScaler()
    df_train_scaled = scaler2.fit_transform(df_train)
    df_test_scaled = scaler2.transform(df_test)

    print(f"\nFitted parameters:")
    print(f"  Mean: {scaler2.mean_}")
    print(f"  Std:  {scaler2.std_}")

    print(f"\nScaled training DataFrame:\n{df_train_scaled}")
    print(f"\nScaled test DataFrame:\n{df_test_scaled}")

    # Test with constant feature
    print("\n" + "="*70)
    print("[Test 3] Handling constant features (std = 0)")
    print("-" * 70)

    X_constant = np.array([
        [1, 5],
        [1, 10],
        [1, 15],
        [1, 20]
    ])

    print(f"Data with constant first column:\n{X_constant}")

    scaler3 = StandardScaler()
    X_constant_scaled = scaler3.fit_transform(X_constant)

    print(f"\nScaled data (first column stays as 0):\n{X_constant_scaled}")
    print(f"\nStd used: {scaler3.std_} (first column std set to 1.0 to avoid division by zero)")

    # Comparison with sklearn
    print("\n" + "="*70)
    print("[Test 4] Comparison with sklearn StandardScaler")
    print("-" * 70)

    try:
        from sklearn.preprocessing import StandardScaler as SklearnScaler

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        # Custom scaler
        custom_scaler = StandardScaler()
        X_custom = custom_scaler.fit_transform(X)

        # Sklearn scaler
        sklearn_scaler = SklearnScaler()
        X_sklearn = sklearn_scaler.fit_transform(X)

        print(f"Original data:\n{X}")
        print(f"\nCustom scaler result:\n{X_custom}")
        print(f"\nSklearn scaler result:\n{X_sklearn}")
        print(f"\nDifference (should be near zero):\n{X_custom - X_sklearn}")
        print(f"\nMax absolute difference: {np.max(np.abs(X_custom - X_sklearn)):.10f}")

    except ImportError:
        print("sklearn not available, skipping comparison")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
