
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from utils.compute_class_weight import compute_class_weight, get_sample_weights
 

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
      - Early stopping
    """

    def __init__(self, layers: List[int], activation: str = 'relu', 
                 output_activation: str = 'sigmoid', random_state: int = 42, momentum=0.0):
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

        self.layers = layers
        self.n_layers = len(layers)
        self.activation = activation
        self.output_activation = output_activation
        self.momentum = momentum

        # Initialize weights and biases
        self.weights = []
        self.biases = []

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
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

        # For SGD with momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]

        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
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
        # Clip to prevent overflow
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
            Input data
        training : bool
            Whether in training mode (affects dropout, batch norm)

        Returns:
        --------
        activations : list
            Activations at each layer
        z_values : list
            Pre-activation values at each layer
        """
        activations = [X]
        z_values = []

        for i in range(self.n_layers - 1):
            # Linear transformation: z = X @ W + b
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)

            # Apply activation
            if i == self.n_layers - 2:  # Output layer
                a = self.activate(z, self.output_activation)
            else:  # Hidden layers
                a = self.activate(z, self.activation)

            activations.append(a)

        return activations, z_values

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        proba : array, shape (n_samples, 1)
            Predicted probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        activations, _ = self.forward(X, training=False)
        return activations[-1]

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        threshold : float
            Classification threshold

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int).ravel()

    # ========================================================================
    # LOSS FUNCTIONS
    # ========================================================================

    def binary_crossentropy(self, y_true, y_pred):
        """
        Binary cross-entropy loss.

        Loss = -[y*log(p) + (1-y)*log(1-p)]
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def binary_cross_entropy_weighted(self, y_true, y_pred, weights):
        """
        Binary cross-entropy loss with sample weights.

        Parameters:
        -----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted probabilities
        weights : ndarray
            Weight for each sample

        Returns:
        --------
        loss : float
            Weighted binary cross-entropy loss
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Calculate loss per sample
        loss_per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Apply weights and average
        weighted_loss = np.mean(weights.reshape(-1, 1) * loss_per_sample)

        return weighted_loss

    # ========================================================================
    # BACKWARD PROPAGATION
    # ========================================================================

    def backward(self, X, y, activations, z_values, sample_weights=None):  
        """
        Backward propagation to compute gradients.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, 1)
            True labels
        activations : list
            Activations from forward pass
        z_values : list
            Pre-activation values from forward pass

        Returns:
        --------
        grad_weights : list
            Gradients for weights
        grad_biases : list
            Gradients for biases
        """
        m = X.shape[0]
        grad_weights = [None] * (self.n_layers - 1)
        grad_biases = [None] * (self.n_layers - 1)

        # Output layer gradient
        # For binary cross-entropy with sigmoid: dL/dz = y_pred - y_true
        if self.output_activation == 'sigmoid':
            delta = activations[-1] - y
        else:
            # General case
            delta = (activations[-1] - y) * self.activate_derivative(z_values[-1], self.output_activation)

        # Apply sample weights if provided
        if sample_weights is not None:
            delta = delta * sample_weights.reshape(-1, 1)


        # Backpropagate through layers
        for i in range(self.n_layers - 2, -1, -1):
            # Gradients for weights and biases
            grad_weights[i] = (activations[i].T @ delta) / m
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:  # Not the first layer
                # Propagate error to previous layer
                delta = (delta @ self.weights[i].T) * self.activate_derivative(z_values[i-1], self.activation)

        return grad_weights, grad_biases

    # ========================================================================
    # OPTIMIZERS
    # ========================================================================

    # Ganti update_weights_sgd yang lama:
    def update_weights_sgd(self, grad_weights, grad_biases, learning_rate):
        """Update weights using SGD with optional momentum."""
        for i in range(len(self.weights)):
            self.velocity_weights[i] = (self.momentum * self.velocity_weights[i] 
                                        - learning_rate * grad_weights[i])
            self.velocity_biases[i] = (self.momentum * self.velocity_biases[i] 
                                    - learning_rate * grad_biases[i])
            self.weights[i] += self.velocity_weights[i]
            self.biases[i] += self.velocity_biases[i]

    def update_weights_adam(self, grad_weights, grad_biases, learning_rate, t, 
                           beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Update weights using Adam optimizer.

        Adam: Adaptive Moment Estimation
        Combines momentum (first moment) and RMSprop (second moment)
        """
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grad_biases[i]

            # Update biased second moment estimate
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (grad_weights[i] ** 2)
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (grad_biases[i] ** 2)

            # Bias correction
            m_weights_corrected = self.m_weights[i] / (1 - beta1 ** t)
            m_biases_corrected = self.m_biases[i] / (1 - beta1 ** t)
            v_weights_corrected = self.v_weights[i] / (1 - beta2 ** t)
            v_biases_corrected = self.v_biases[i] / (1 - beta2 ** t)

            # Update weights
            self.weights[i] -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon)
            self.biases[i] -= learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + epsilon)

    def update_weights_adabelief(self, grad_weights, grad_biases, learning_rate, t, 
                                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Update weights using AdaBelief optimizer.

        AdaBelief: Adapting Stepsizes by the Belief in Observed Gradients

        Key difference from Adam:
        - Adam: v_t = β₂*v + (1-β₂)*g²
        - AdaBelief: s_t = β₂*s + (1-β₂)*(g - m)²
        """
        # Initialize AdaBelief-specific variables if not exists
        if not hasattr(self, 's_weights'):
            self.s_weights = [np.zeros_like(w) for w in self.weights]
            self.s_biases = [np.zeros_like(b) for b in self.biases]

        for i in range(len(self.weights)):
            # Update first moment (same as Adam)
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_weights[i]
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grad_biases[i]

            # Update second moment (AdaBelief: uses gradient prediction error)
            grad_diff_w = grad_weights[i] - self.m_weights[i]
            grad_diff_b = grad_biases[i] - self.m_biases[i]

            self.s_weights[i] = beta2 * self.s_weights[i] + (1 - beta2) * (grad_diff_w ** 2)
            self.s_biases[i] = beta2 * self.s_biases[i] + (1 - beta2) * (grad_diff_b ** 2)

            # Bias correction
            m_weights_corrected = self.m_weights[i] / (1 - beta1 ** t)
            m_biases_corrected = self.m_biases[i] / (1 - beta1 ** t)
            s_weights_corrected = self.s_weights[i] / (1 - beta2 ** t)
            s_biases_corrected = self.s_biases[i] / (1 - beta2 ** t)

            # Update weights
            self.weights[i] -= learning_rate * m_weights_corrected / (np.sqrt(s_weights_corrected + epsilon) + epsilon)
            self.biases[i] -= learning_rate * m_biases_corrected / (np.sqrt(s_biases_corrected + epsilon) + epsilon)

    # ========================================================================
    # TRAINING
    # ========================================================================

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32,
            learning_rate=0.01, optimizer='sgd', verbose=1, 
            early_stopping_patience=None, class_weight=None):
        """
        Train the neural network.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, 1)
            Training labels
        X_val : array-like, optional
            Validation data
        y_val : array-like, optional
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch gradient descent
        learning_rate : float
            Learning rate
        optimizer : str
            Optimizer to use ('sgd' or 'adam' or 'adabelief')
        verbose : int
            0=silent, 1=progress bar, 2=one line per epoch
        early_stopping_patience : int, optional
            Stop if validation loss doesn't improve for this many epochs
        class_weight : dict or 'balanced' or None
            Weights for each class. If 'balanced', automatically compute
            balanced weights. If dict, should map class indices to weights.
            If None, all classes have weight 1.0 (default).

        Returns:
        --------
        self : object
            Fitted neural network
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values

        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Handle class weights
        sample_weights = None
        if class_weight is not None:
            classes = np.unique(y)
            class_weight_dict = compute_class_weight(class_weight, classes, y)
            sample_weights = get_sample_weights(y, class_weight_dict)

            if verbose >= 1:
                print(f"Class weights: {class_weight_dict}")

        # Validation data
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.DataFrame) or isinstance(y_val, pd.Series):
                y_val = y_val.values
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        # Early stopping
        best_val_loss = np.inf
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                activations, z_values = self.forward(X_batch, training=True)

                # Compute loss (with sample weights if provided)
                if sample_weights is not None:
                    weights_batch = sample_weights[indices[start_idx:end_idx]]
                    batch_loss = self.binary_cross_entropy_weighted(y_batch, activations[-1], weights_batch)
                else:
                    batch_loss = self.binary_crossentropy(y_batch, activations[-1])
                epoch_loss += batch_loss


                # Backward pass (with sample weights if provided)
                if sample_weights is not None:
                    weights_batch = sample_weights[indices[start_idx:end_idx]]
                    grad_weights, grad_biases = self.backward(X_batch, y_batch, activations, z_values, weights_batch)
                else:
                    grad_weights, grad_biases = self.backward(X_batch, y_batch, activations, z_values)

                # Update weights
                if optimizer == 'sgd':
                    self.update_weights_sgd(grad_weights, grad_biases, learning_rate)
                elif optimizer == 'adam':
                    t = epoch * n_batches + batch + 1
                    self.update_weights_adam(grad_weights, grad_biases, learning_rate, t)
                elif optimizer == 'adabelief':
                    t = epoch * n_batches + batch + 1
                    self.update_weights_adabelief(grad_weights, grad_biases, learning_rate, t)

            # Average loss for epoch
            epoch_loss /= n_batches

            # Calculate accuracy
            y_pred_train = self.predict(X)
            train_acc = np.mean(y_pred_train == y.ravel())

            # Store history
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(train_acc)

            # Validation
            if has_validation:
                val_activations, _ = self.forward(X_val, training=False)
                val_loss = self.binary_crossentropy(y_val, val_activations[-1])
                y_pred_val = self.predict(X_val)
                val_acc = np.mean(y_pred_val == y_val.ravel())

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose > 0:
                                print(f"\nEarly stopping at epoch {epoch+1}")
                            break

            # Print progress
            if verbose == 2:
                if has_validation:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - "
                          f"acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {train_acc:.4f}")

        self.is_trained = True
        return self

    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the model on test data.

        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            Test labels
        threshold : float
            Classification threshold

        Returns:
        --------
        results : dict
            Dictionary containing loss and accuracy
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Predictions
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X, threshold)

        # Metrics
        loss = self.binary_crossentropy(y, y_pred_proba)
        accuracy = np.mean(y_pred == y.ravel())

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layers}, activation='{self.activation}', trained={self.is_trained})"

# ================================================================================
# KEY CONCEPTS
# ================================================================================

# SGD: θ = θ - α*g
#   • Basic gradient descent
#   • No adaptive learning rate

# Adam: v = β₂*v + (1-β₂)*g²
#   • Adapts by gradient variance
#   • Fast convergence

# AdaBelief: s = β₂*s + (1-β₂)*(g-m)²
#   • Adapts by gradient prediction error
#   • Better generalization
#   • More stable training

# The key difference: AdaBelief uses (g-m)² instead of g²
# This measures how much the gradient differs from its prediction!

# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NEURAL NETWORK FROM SCRATCH - DEMONSTRATION")
    print("="*70)

    # Generate synthetic data
    print("\n[Test 1] Binary Classification on Synthetic Data")
    print("-" * 70)

    np.random.seed(42)

    # Generate data (simple XOR-like problem)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Create and train network
    print("\nTraining neural network (2 -> 16 -> 8 -> 1)...")

    nn = NeuralNetwork(layers=[2, 16, 8, 1], activation='relu', random_state=42)
    nn.fit(X_train, y_train, X_val=X_test, y_val=y_test, 
           epochs=50, batch_size=32, learning_rate=0.01, 
           optimizer='adam', verbose=2)

    # Evaluate
    results = nn.evaluate(X_test, y_test)
    print(f"\nTest Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
