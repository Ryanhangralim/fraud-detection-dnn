import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from utils.compute_class_weight import compute_class_weight, get_sample_weights
from utils.metrics import average_precision_score

class NeuralNetwork:
    """
    Feedforward Neural Network implementation from scratch.

    Supports:
    - Multiple hidden layers
    - Activation functions: relu, sigmoid, tanh
    - Binary cross-entropy loss
    - Optimizers: SGD, Adam, AdaBelief
    - Batch normalization
    - Dropout
    - Early stopping (val_loss  OR  val_pr_auc)
    """

    def __init__(self, layers: List[int], activation: str = 'relu',
                 output_activation: str = 'sigmoid', random_state: int = 42):
        """
        Initialize neural network architecture.

        Parameters:
        -----------
        layers : List[int]
            Number of neurons in each layer [input, hidden1, hidden2, ..., output]
            Example: [30, 128, 64, 32, 1] for input=30, 3 hidden layers, output=1
        activation : str
            Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
        output_activation : str
            Activation function for output layer ('sigmoid', 'softmax')
        random_state : int
            Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_state)

        self.layers          = layers
        self.n_layers        = len(layers)
        self.activation      = activation
        self.output_activation = output_activation

        # Initialize weights and biases
        self.weights = []
        self.biases  = []

        for i in range(len(layers) - 1):
            if activation == 'relu':
                w = self.rng.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            else:
                w = self.rng.randn(layers[i], layers[i+1]) * np.sqrt(1.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # For Adam and AdaBelief optimizer
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases  = [np.zeros_like(b) for b in self.biases]
        self.v_biases  = [np.zeros_like(b) for b in self.biases]

        # For SGD with momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases  = [np.zeros_like(b) for b in self.biases]

        # Training history
        self.history = {
            'loss'        : [],
            'val_loss'    : [],
            'accuracy'    : [],
            'val_accuracy': [],
            'val_pr_auc'  : [],   # NEW: tracked every epoch when val data present
        }

        self.is_trained = False

    # ========================================================================
    # ACTIVATION FUNCTIONS
    # ========================================================================

    def relu(self, x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation: 1 / (1 + e^(-x))"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2

    def activate(self, x, activation=None):
        """Apply activation function"""
        if activation is None:
            activation = self.activation
        if activation == 'relu':
            return self.relu(x)
        elif activation == 'sigmoid':
            return self.sigmoid(x)
        elif activation == 'tanh':
            return self.tanh(x)
        else:
            return x

    def activate_derivative(self, x, activation=None):
        """Apply activation derivative"""
        if activation is None:
            activation = self.activation
        if activation == 'relu':
            return self.relu_derivative(x)
        elif activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif activation == 'tanh':
            return self.tanh_derivative(x)
        else:
            return np.ones_like(x)

    # ========================================================================
    # FORWARD PROPAGATION
    # ========================================================================

    def forward(self, X, training=True):
        """
        Forward propagation through the network.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        training : bool
            Whether in training mode (affects dropout, batch norm)

        Returns:
        --------
        activations : list
        z_values    : list
        """
        activations = [X]
        z_values    = []

        for i in range(self.n_layers - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)

            if i == self.n_layers - 2:       # Output layer
                a = self.activate(z, self.output_activation)
            else:                             # Hidden layers
                a = self.activate(z, self.activation)

            activations.append(a)

        return activations, z_values

    def predict_proba(self, X):
        """Predict class probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        activations, _ = self.forward(X, training=False)
        return activations[-1]

    def predict(self, X, threshold=0.5):
        """Predict binary class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int).ravel()

    # ========================================================================
    # LOSS FUNCTIONS
    # ========================================================================

    def binary_crossentropy(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_cross_entropy_weighted(self, y_true, y_pred, weights):
        """Binary cross-entropy loss with sample weights."""
        epsilon = 1e-15
        y_pred  = np.clip(y_pred, epsilon, 1 - epsilon)
        loss_per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(weights.reshape(-1, 1) * loss_per_sample)

    # ========================================================================
    # BACKWARD PROPAGATION
    # ========================================================================

    def backward(self, X, y, activations, z_values, sample_weights=None):
        """Backward propagation to compute gradients."""
        m = X.shape[0]
        grad_weights = [None] * (self.n_layers - 1)
        grad_biases  = [None] * (self.n_layers - 1)

        if self.output_activation == 'sigmoid':
            delta = activations[-1] - y
        else:
            delta = (activations[-1] - y) * self.activate_derivative(
                z_values[-1], self.output_activation)

        if sample_weights is not None:
            delta = delta * sample_weights.reshape(-1, 1)

        for i in range(self.n_layers - 2, -1, -1):
            grad_weights[i] = (activations[i].T @ delta) / m
            grad_biases[i]  = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activate_derivative(
                    z_values[i-1], self.activation)

        return grad_weights, grad_biases

    # ========================================================================
    # OPTIMIZERS
    # ========================================================================

    def update_weights_sgd(self, grad_weights, grad_biases, learning_rate, momentum=0.0):
        """Update weights using SGD with optional momentum."""
        for i in range(len(self.weights)):
            self.velocity_weights[i] = (momentum * self.velocity_weights[i]
                                        - learning_rate * grad_weights[i])
            self.velocity_biases[i]  = (momentum * self.velocity_biases[i]
                                        - learning_rate * grad_biases[i])
            self.weights[i] += self.velocity_weights[i]
            self.biases[i]  += self.velocity_biases[i]

    def update_weights_adam(self, grad_weights, grad_biases, learning_rate, t,
                            beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update weights using Adam optimizer."""
        for i in range(len(self.weights)):
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
            self.m_biases[i]  = beta1 * self.m_biases[i]  + (1 - beta1) * grad_biases[i]
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (grad_weights[i] ** 2)
            self.v_biases[i]  = beta2 * self.v_biases[i]  + (1 - beta2) * (grad_biases[i] ** 2)

            m_w_corr = self.m_weights[i] / (1 - beta1 ** t)
            m_b_corr = self.m_biases[i]  / (1 - beta1 ** t)
            v_w_corr = self.v_weights[i] / (1 - beta2 ** t)
            v_b_corr = self.v_biases[i]  / (1 - beta2 ** t)

            self.weights[i] -= learning_rate * m_w_corr / (np.sqrt(v_w_corr) + epsilon)
            self.biases[i]  -= learning_rate * m_b_corr / (np.sqrt(v_b_corr) + epsilon)

    def update_weights_adabelief(self, grad_weights, grad_biases, learning_rate, t,
                                  beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Update weights using AdaBelief optimizer.

        Key difference from Adam:
        - Adam:     v_t = β₂*v + (1-β₂)*g²
        - AdaBelief: s_t = β₂*s + (1-β₂)*(g - m)²
        """
        if not hasattr(self, 's_weights'):
            self.s_weights = [np.zeros_like(w) for w in self.weights]
            self.s_biases  = [np.zeros_like(b) for b in self.biases]

        for i in range(len(self.weights)):
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
            self.m_biases[i]  = beta1 * self.m_biases[i]  + (1 - beta1) * grad_biases[i]

            grad_diff_w = grad_weights[i] - self.m_weights[i]
            grad_diff_b = grad_biases[i]  - self.m_biases[i]

            self.s_weights[i] = beta2 * self.s_weights[i] + (1 - beta2) * (grad_diff_w ** 2)
            self.s_biases[i]  = beta2 * self.s_biases[i]  + (1 - beta2) * (grad_diff_b ** 2)

            m_w_corr = self.m_weights[i] / (1 - beta1 ** t)
            m_b_corr = self.m_biases[i]  / (1 - beta1 ** t)
            s_w_corr = self.s_weights[i] / (1 - beta2 ** t)
            s_b_corr = self.s_biases[i]  / (1 - beta2 ** t)

            self.weights[i] -= learning_rate * m_w_corr / (np.sqrt(s_w_corr + epsilon) + epsilon)
            self.biases[i]  -= learning_rate * m_b_corr / (np.sqrt(s_b_corr + epsilon) + epsilon)

    # ========================================================================
    # WEIGHT SNAPSHOT HELPERS  (for best-weight restore)
    # ========================================================================

    def _get_weights_snapshot(self):
        """Return a deep copy of current weights and biases."""
        return (
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases],
        )

    def _restore_weights(self, snapshot):
        """Restore weights and biases from a snapshot."""
        best_w, best_b = snapshot
        self.weights = [w.copy() for w in best_w]
        self.biases  = [b.copy() for b in best_b]

    # ========================================================================
    # TRAINING
    # ========================================================================

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32,
            learning_rate=0.01, optimizer='sgd', momentum=0.0, verbose=1,
            early_stopping_patience=None, early_stopping_monitor='val_pr_auc',
            class_weight=None):
        """
        Train the neural network.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) or (n_samples, 1)
        X_val : array-like, optional
        y_val : array-like, optional
        epochs : int
        batch_size : int
        learning_rate : float
        optimizer : str  ('sgd' | 'adam' | 'adabelief')
        verbose : int    (0=silent, 1=progress bar, 2=one line per epoch)
        early_stopping_patience : int, optional
        early_stopping_monitor : str
            Metric to watch for early stopping.
            'val_pr_auc'  – stop when Val PR-AUC stops INCREASING  (default)
            'val_loss'    – stop when Val Loss stops DECREASING     (legacy)
        class_weight : dict | 'balanced' | None

        Returns:
        --------
        self
        """
        # ── Convert inputs ────────────────────────────────────────────────
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # ── Class weights ─────────────────────────────────────────────────
        sample_weights = None
        if class_weight is not None:
            classes = np.unique(y)
            class_weight_dict = compute_class_weight(class_weight, classes, y)
            sample_weights    = get_sample_weights(y, class_weight_dict)
            if verbose >= 1:
                print(f"Class weights: {class_weight_dict}")

        # ── Validation data ───────────────────────────────────────────────
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, (pd.DataFrame, pd.Series)):
                y_val = y_val.values
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        # ── Early stopping setup ──────────────────────────────────────────
        monitor      = early_stopping_monitor   # 'val_pr_auc' or 'val_loss'
        maximize     = (monitor == 'val_pr_auc') # True → want higher value
        best_monitor = -np.inf if maximize else np.inf
        patience_counter  = 0
        best_weights_snap = None                # will hold best-epoch snapshot

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(epochs):
            indices    = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0

            for batch in range(n_batches):
                s = batch * batch_size
                e = min(s + batch_size, n_samples)

                X_batch = X_shuffled[s:e]
                y_batch = y_shuffled[s:e]

                activations, z_values = self.forward(X_batch, training=True)

                if sample_weights is not None:
                    w_batch    = sample_weights[indices[s:e]]
                    batch_loss = self.binary_cross_entropy_weighted(
                        y_batch, activations[-1], w_batch)
                else:
                    batch_loss = self.binary_crossentropy(y_batch, activations[-1])
                epoch_loss += batch_loss

                if sample_weights is not None:
                    w_batch = sample_weights[indices[s:e]]
                    grad_weights, grad_biases = self.backward(
                        X_batch, y_batch, activations, z_values, w_batch)
                else:
                    grad_weights, grad_biases = self.backward(
                        X_batch, y_batch, activations, z_values)

                t = epoch * n_batches + batch + 1
                if optimizer == 'sgd':
                    self.update_weights_sgd(grad_weights, grad_biases, learning_rate, momentum)
                elif optimizer == 'adam':
                    self.update_weights_adam(grad_weights, grad_biases, learning_rate, t)
                elif optimizer == 'adabelief':
                    self.update_weights_adabelief(grad_weights, grad_biases, learning_rate, t)

            epoch_loss  /= n_batches
            y_pred_train = self.predict(X)
            train_acc    = np.mean(y_pred_train == y.ravel())

            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(train_acc)

            # ── Validation ────────────────────────────────────────────────
            if has_validation:
                val_act, _ = self.forward(X_val, training=False)
                val_loss   = self.binary_crossentropy(y_val, val_act[-1])
                y_pred_val = self.predict(X_val)
                val_acc    = np.mean(y_pred_val == y_val.ravel())

                # Compute Val PR-AUC
                val_proba  = val_act[-1].ravel()
                val_pr_auc = average_precision_score(y_val.ravel(), val_proba)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                self.history['val_pr_auc'].append(val_pr_auc)

                # ── Early stopping ────────────────────────────────────────
                if early_stopping_patience is not None:
                    current = val_pr_auc if monitor == 'val_pr_auc' else val_loss
                    improved = (current > best_monitor) if maximize else (current < best_monitor)

                    if improved:
                        best_monitor      = current
                        patience_counter  = 0
                        best_weights_snap = self._get_weights_snapshot()
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose > 0:
                                print(f"\nEarly stopping at epoch {epoch+1} "
                                      f"(best {monitor}={best_monitor:.4f})")
                            # Restore the best weights before returning
                            if best_weights_snap is not None:
                                self._restore_weights(best_weights_snap)
                            break

            # ── Verbose output ────────────────────────────────────────────
            if verbose == 2:
                if has_validation:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} "
                          f"- acc: {train_acc:.4f} - val_loss: {val_loss:.4f} "
                          f"- val_acc: {val_acc:.4f} - val_pr_auc: {val_pr_auc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} "
                          f"- acc: {train_acc:.4f}")

        self.is_trained = True
        return self

    # ========================================================================
    # EVALUATION
    # ========================================================================

    def evaluate(self, X, y, threshold=0.5):
        """Evaluate the model on test data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_pred_proba = self.predict_proba(X)
        y_pred       = self.predict(X, threshold)
        loss         = self.binary_crossentropy(y, y_pred_proba)
        accuracy     = np.mean(y_pred == y.ravel())

        return {'loss': loss, 'accuracy': accuracy}

    def __repr__(self):
        return (f"NeuralNetwork(layers={self.layers}, "
                f"activation='{self.activation}', trained={self.is_trained})")


# ================================================================================
# KEY CONCEPTS
# ================================================================================

# SGD:       θ = θ - α*g
# Adam:      v = β₂*v + (1-β₂)*g²           (adapts by gradient variance)
# AdaBelief: s = β₂*s + (1-β₂)*(g-m)²      (adapts by gradient prediction error)

# Early stopping monitors:
#   val_pr_auc → MAXIMIZE  (stop when PR-AUC on val set stops improving)
#   val_loss   → MINIMIZE  (legacy behaviour)
# Best weights at the peak epoch are RESTORED automatically on stop.


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NEURAL NETWORK FROM SCRATCH - DEMONSTRATION")
    print("="*70)

    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Class distribution: {np.bincount(y_train)}")

    print("\nTraining with early_stopping_monitor='val_pr_auc'...")
    nn = NeuralNetwork(layers=[2, 16, 8, 1], activation='relu', random_state=42)
    nn.fit(X_train, y_train, X_val=X_test, y_val=y_test,
           epochs=100, batch_size=32, learning_rate=0.01,
           optimizer='adabelief', verbose=2,
           early_stopping_patience=10,
           early_stopping_monitor='val_pr_auc')

    results = nn.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"  Loss     : {results['loss']:.4f}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Best val PR-AUC during training: {max(nn.history['val_pr_auc']):.4f}")
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
