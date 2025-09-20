import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize network parameters (weights)
np.random.seed(42)
w1, w2 = np.random.randn(), np.random.randn()  # Weights for input to hidden layer
v1, v2 = np.random.randn(), np.random.randn()  # Weights for hidden to output layer

# Learning rate
eta = 0.1

# Training data: x (input) and t (target)
X = np.array([0.5, 1.0, 1.5, 2.0])  # Input values
T = np.array([0.2, 0.5, 0.7, 0.9])  # Target values (desired outputs)

# Training loop
epochs = 5000
for epoch in range(epochs):
    total_loss = 0
    
    for x, t in zip(X, T):
        # === Forward Pass ===
        k1 = w1 * x
        k2 = w2 * x
        h1 = sigmoid(k1)
        h2 = sigmoid(k2)
        y = v1 * h1 + v2 * h2  # Output

        # Compute loss (Mean Squared Error)
        loss = 0.5 * (y - t) ** 2
        total_loss += loss

        # === Backpropagation ===
        d_loss_dy = (y - t)  # Derivative of loss w.r.t. output
        d_y_dv1 = h1  # Derivative of y w.r.t. v1
        d_y_dv2 = h2  # Derivative of y w.r.t. v2
        
        d_y_dh1 = v1  # Derivative of y w.r.t. h1
        d_y_dh2 = v2  # Derivative of y w.r.t. h2

        d_h1_dk1 = sigmoid_derivative(h1)  # Derivative of h1 w.r.t. k1
        d_h2_dk2 = sigmoid_derivative(h2)  # Derivative of h2 w.r.t. k2

        d_k1_dw1 = x  # Derivative of k1 w.r.t. w1
        d_k2_dw2 = x  # Derivative of k2 w.r.t. w2

        # Compute weight updates using chain rule
        d_loss_dw1 = d_loss_dy * d_y_dh1 * d_h1_dk1 * d_k1_dw1
        d_loss_dw2 = d_loss_dy * d_y_dh2 * d_h2_dk2 * d_k2_dw2
        d_loss_dv1 = d_loss_dy * d_y_dv1
        d_loss_dv2 = d_loss_dy * d_y_dv2

        # === Update Weights (Gradient Descent) ===
        w1 -= eta * d_loss_dw1
        w2 -= eta * d_loss_dw2
        v1 -= eta * d_loss_dv1
        v2 -= eta * d_loss_dv2

    # Print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.6f}")

# Final weights
print("\nFinal Weights:")
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, v1 = {v1:.4f}, v2 = {v2:.4f}")
