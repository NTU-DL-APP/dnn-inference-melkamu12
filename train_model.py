import tensorflow as tf
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

# === Config ===
MODEL_NAME = 'fashion_mnist'
H5_PATH = f'{MODEL_NAME}.h5'
NPZ_PATH = f'model/{MODEL_NAME}.npz'
JSON_PATH = f'model/{MODEL_NAME}.json'

# === Ensure model directory exists ===
os.makedirs("model", exist_ok=True)

# === Step 1: Load Data Using only 't10k' and split ===
from utils.mnist_reader import load_mnist
from tensorflow.keras.utils import to_categorical

print("📥 Loading Fashion-MNIST from 't10k' gzip file...")
x_all, y_all = load_mnist('./data/fashion', kind='t10k')

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.20, random_state=42)

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"🟢 Training samples: {len(x_train)}, Test samples: {len(x_test)}")

# === Step 2: Build the Model ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])


# === Step 3: Compile & Train ===
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

# === Step 4: Evaluate ===
loss, acc = model.evaluate(x_test, y_test)
print(f"✅ Test accuracy: {acc:.4f}")

# === Step 5: Save .h5 Model ===
model.save(H5_PATH)
print(f"📦 Saved model to: {H5_PATH}")

# === Step 6: Export Weights to .npz ===
params = {}
for layer in model.layers:
    weights = layer.get_weights()
    for i, w in enumerate(weights):
        params[f"{layer.name}_{i}"] = w
np.savez(NPZ_PATH, **params)
print(f"📦 Exported weights to: {NPZ_PATH}")

# === Step 7: Export Architecture to .json ===
arch = []
for layer in model.layers:
    config = layer.get_config()
    info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": config,
        "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
    }
    arch.append(info)

with open(JSON_PATH, "w") as f:
    json.dump(arch, f, indent=2)
print(f"📦 Exported architecture to: {JSON_PATH}")
