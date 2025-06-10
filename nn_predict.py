import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
    return exp / np.sum(exp, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense Layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward Pass ===
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        ltype = layer["type"]
        cfg = layer["config"]
        wnames = layer["weights"]

        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

        elif ltype == "Dropout":
            # Dropout is ignored during inference
            continue

    return x

# === Inference wrapper ===
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)

# === Example usage ===
if __name__ == "__main__":
    # Load exported weights and architecture
    weights = np.load("model/fashion_mnist.npz")
    with open("model/fashion_mnist.json") as f:
        model_arch = json.load(f)

    # Load sample data (e.g. from t10k for verification)
    from utils.mnist_reader import load_mnist
    x_all, y_all = load_mnist('./data/fashion', kind='t10k')

    # Normalize and reshape input
    x_all = x_all.astype(np.float32) / 255.0
    x_all = x_all.reshape(x_all.shape[0], 28, 28)  # in case it's flat
    y_true = y_all

    # Run inference
    y_pred = nn_inference(model_arch, weights, x_all)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Accuracy check
    acc = np.mean(y_pred_classes == y_true)
    print(f"✅ NumPy Inference Accuracy: {acc:.4f}")
