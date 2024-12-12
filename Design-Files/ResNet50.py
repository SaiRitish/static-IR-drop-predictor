from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import UpSampling2D, Concatenate, Conv2D, BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Debugging helper
def debug(message):
    print(f"[DEBUG] {message}")

# --- Preprocessing ---
def preprocess_csv(file_path, target_shape=(128, 128)):
    data = pd.read_csv(file_path, header=None).to_numpy(dtype=np.float32)
    data = tf.image.resize(data[..., np.newaxis], target_shape, method="bilinear").numpy()
    max_val = np.max(data)
    return data / max_val if max_val > 0 else data

def load_data(base_path):
    available_current_files = [
        os.path.join(base_path, file)
        for file in os.listdir(base_path)
        if "_current" in file and file.endswith(".csv") and "current_map" in file
    ]
    available_pdn_files = [
        os.path.join(base_path, file)
        for file in os.listdir(base_path)
        if "pdn_density" in file and file.endswith(".csv")
    ]
    available_dist_files = [
        os.path.join(base_path, file)
        for file in os.listdir(base_path)
        if "eff_dist" in file and file.endswith(".csv")
    ]
    available_ir_drop_files = [
        os.path.join(base_path, file)
        for file in os.listdir(base_path)
        if "ir_drop" in file and file.endswith(".csv") and "current_map" in file
    ]

    def extract_index(file_name):
        return int(file_name.split("map")[1].split("_")[0])

    available_current_files.sort(key=lambda f: extract_index(f))
    available_pdn_files.sort(key=lambda f: extract_index(f))
    available_dist_files.sort(key=lambda f: extract_index(f))
    available_ir_drop_files.sort(key=lambda f: extract_index(f))

    # Preprocess data
    input_currents = np.array([preprocess_csv(file) for file in available_current_files])
    input_pdns = np.array([preprocess_csv(file) for file in available_pdn_files])
    input_dists = np.array([preprocess_csv(file) for file in available_dist_files])
    output_images = np.array([preprocess_csv(file) for file in available_ir_drop_files])

    return input_currents, input_pdns, input_dists, output_images

# --- ResNet Model ---
def resnet_based_model(input_shape=(128, 128, 1)):
    # Inputs
    input_current = Input(shape=input_shape, name="current_input")
    input_pdn = Input(shape=input_shape, name="pdn_input")
    input_dist = Input(shape=input_shape, name="dist_input")

    # Combine inputs
    merged_inputs = Concatenate()([input_current, input_pdn, input_dist])

    # Convert to 3 channels for ResNet compatibility
    merged_inputs_3c = Conv2D(3, (1, 1), activation="relu", padding="same")(merged_inputs)

    # ResNet Backbone
    base_model = ResNet50(weights=None, include_top=False, input_shape=(128, 128, 3))
    resnet_features = base_model(merged_inputs_3c)

    # Decoder
    u6 = UpSampling2D((2, 2))(resnet_features)  # Upsample from (4x4) to (8x8)
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.4)(c6)

    u7 = UpSampling2D((2, 2))(c6)  # Upsample from (8x8) to (16x16)
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.3)(c7)

    u8 = UpSampling2D((2, 2))(c7)  # Upsample from (16x16) to (32x32)
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)

    u9 = UpSampling2D((2, 2))(c8)  # Upsample from (32x32) to (64x64)
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)

    u10 = UpSampling2D((2, 2))(c9)  # Upsample from (64x64) to (128x128)
    c10 = Conv2D(32, (3, 3), activation="relu", padding="same")(u10)
    c10 = BatchNormalization()(c10)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c10)

    model = Model(inputs=[input_current, input_pdn, input_dist], outputs=outputs)
    return model

# --- Main ---
# Set paths
base_path = "/content/drive/My Drive/ML-for-IR-drop-main/benchmarks/fake-circuit-data"
output_dir = "/content/drive/My Drive/ML-for-IR-drop-results"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

input_currents, input_pdns, input_dists, output_images = load_data(base_path)

# Split data
split_idx = len(input_currents) - 10
train_currents, test_currents = input_currents[:split_idx], input_currents[split_idx:]
train_pdns, test_pdns = input_pdns[:split_idx], input_pdns[split_idx:]
train_dists, test_dists = input_dists[:split_idx], input_dists[split_idx:]
train_outputs, test_outputs = output_images[:split_idx], output_images[split_idx:]

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices(((train_currents, train_pdns, train_dists), train_outputs)).batch(16)
test_ds = tf.data.Dataset.from_tensor_slices(((test_currents, test_pdns, test_dists), test_outputs)).batch(16)

# Define model
model = resnet_based_model(input_shape=(128, 128, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])

# Train model
history = model.fit(train_ds, validation_data=test_ds, epochs=500)

# Plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Generate predictions
predictions = model.predict(test_ds)

# Visualize predictions with MAE
for i in range(len(test_currents)):
    # Calculate MAE for the current test case
    mae = np.mean(np.abs(test_outputs[i].squeeze() - predictions[i].squeeze()))
    
    # Plot inputs, ground truth, and prediction
    plt.figure(figsize=(20, 5))
    
    # Input Current Map
    plt.subplot(1, 5, 1)
    plt.title("Input Current")
    plt.imshow(test_currents[i].squeeze(), cmap="viridis")
    plt.colorbar()
    
    # Input PDN Density Map
    plt.subplot(1, 5, 2)
    plt.title("Input PDN")
    plt.imshow(test_pdns[i].squeeze(), cmap="viridis")
    plt.colorbar()
    
    # Input Effective Distance
    plt.subplot(1, 5, 3)
    plt.title("Input Effective Distance")
    plt.imshow(test_dists[i].squeeze(), cmap="viridis")
    plt.colorbar()
    
    # Ground Truth IR Drop Map
    plt.subplot(1, 5, 4)
    plt.title("Ground Truth")
    plt.imshow(test_outputs[i].squeeze(), cmap="viridis")
    plt.colorbar()
    
    # Predicted IR Drop Map with MAE in Title
    plt.subplot(1, 5, 5)
    plt.title(f"Prediction\n(MAE: {mae:.4f})")
    plt.imshow(predictions[i].squeeze(), cmap="viridis")
    plt.colorbar()
    
    # Save the figure to the output directory
    plt.savefig(os.path.join(output_dir, f"prediction_case_{i+1}.png"))
    
    # Display the figure
    plt.show()
