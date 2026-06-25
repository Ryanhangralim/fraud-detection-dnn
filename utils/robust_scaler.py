import numpy as np
import pandas as pd


class RobustScaler:
    """
    Custom RobustScaler dari scratch.
    Menggunakan median dan IQR untuk scaling — robust terhadap outlier.
    Compatible dengan sklearn API: fit, transform, fit_transform, inverse_transform.
    """

    def __init__(self, quantile_range=(25.0, 75.0)):
        """
        Parameters
        ----------
        quantile_range : tuple (q_min, q_max), default (25.0, 75.0)
            Rentang persentil untuk menghitung IQR.
        """
        self.quantile_range = quantile_range
        self.center_ = None   # median per fitur
        self.scale_  = None   # IQR per fitur

    def fit(self, X):
        """
        Hitung median dan IQR dari data training.

        Parameters
        ----------
        X : array-like atau DataFrame, shape (n_samples, n_features)
        """
        X_arr = self._to_numpy(X)

        q_min, q_max = self.quantile_range
        self.center_ = np.median(X_arr, axis=0)
        self.scale_  = np.percentile(X_arr, q_max, axis=0) - \
                       np.percentile(X_arr, q_min, axis=0)

        # Hindari pembagian nol: jika IQR = 0, set scale = 1
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

        return self

    def transform(self, X):
        """
        Terapkan scaling: (X - median) / IQR

        Parameters
        ----------
        X : array-like atau DataFrame, shape (n_samples, n_features)

        Returns
        -------
        X_scaled : np.ndarray
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler belum di-fit. Panggil fit() terlebih dahulu.")

        X_arr = self._to_numpy(X)
        return (X_arr - self.center_) / self.scale_

    def fit_transform(self, X):
        """Fit lalu transform sekaligus."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Kembalikan data ke skala asli: X = X_scaled * IQR + median

        Parameters
        ----------
        X_scaled : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_original : np.ndarray
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler belum di-fit.")

        X_arr = self._to_numpy(X_scaled)
        return X_arr * self.scale_ + self.center_

    def _to_numpy(self, X):
        """Helper: konversi DataFrame atau list ke numpy array float64."""
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.array(X, dtype=np.float64)