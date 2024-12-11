
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

# Debugging helper
def debug(message):
    print(f"[DEBUG] {message}")

# --- Preprocessing ---
base_path = "/content/drive/My Drive/ML-for-IR-drop/benchmarks/fake-circuit-data/"

# Dynamically find files
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

# Sort by numeric indices
def extract_index(file_name):
    return int(file_name.split("map")[1].split("_")[0])

available_current_files.sort(key=lambda f: extract_index(f))
available_pdn_files.sort(key=lambda f: extract_index(f))
available_dist_files.sort(key=lambda f: extract_index(f))
available_ir_drop_files.sort(key=lambda f: extract_index(f))

# Ensure alignment of files
if not (len(available_current_files) == len(available_pdn_files) == len(available_ir_drop_files)):
    raise ValueError(
        f"Mismatch in file counts: {len(available_current_files)} currents, {len(available_pdn_files)} PDNs, {len(available_dist_files)} distancess, {len(available_ir_drop_files)} IR drops."
    )

debug(f"Found {len(available_current_files)} current, {len(available_pdn_files)} PDN density, {len(available_dist_files)} effective distance, and {len(available_ir_drop_files)} IR drop files.")

# Preprocess CSV files
def preprocess_csv(file_path, target_shape=(128, 128)):
    debug(f"Processing file: {file_path}")
    data = pd.read_csv(file_path, header=None).to_numpy(dtype=np.float32)
    data = tf.image.resize(data[..., np.newaxis], target_shape, method="bilinear").numpy()
    max_val = np.max(data)
    return data / max_val if max_val > 0 else data

# Load and preprocess data
input_currents = np.array([preprocess_csv(file) for file in available_current_files])
input_pdns = np.array([preprocess_csv(file) for file in available_pdn_files])
input_dists = np.array([preprocess_csv(file) for file in available_dist_files])
output_images = np.array([preprocess_csv(file) for file in available_ir_drop_files])

# --- Data Splitting ---
debug("Splitting data...")
split_idx = len(input_currents) - 10  # Train on all images, test on the last 10
train_currents, test_currents = input_currents[:split_idx], input_currents[split_idx:]
train_pdns, test_pdns = input_pdns[:split_idx], input_pdns[split_idx:]
train_dists, test_dists = input_dists[:split_idx], input_dists[split_idx:]
train_outputs, test_outputs = output_images[:split_idx], output_images[split_idx:]

debug(f"Training size: {len(train_currents)}, Testing size: {len(test_currents)}")

# --- Perceptual Loss ---
# Load VGG19 model
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
vgg.trainable = False

# Select a specific layer's output for perceptual loss
perceptual_layer = vgg.get_layer("block5_conv4").output
perceptual_model = Model(inputs=vgg.input, outputs=perceptual_layer)

def perceptual_loss(y_true, y_pred):
    y_true_rgb = tf.image.grayscale_to_rgb(y_true)
    y_pred_rgb = tf.image.grayscale_to_rgb(y_pred)
    true_features = perceptual_model(y_true_rgb)
    pred_features = perceptual_model(y_pred_rgb)
    return tf.reduce_mean(tf.square(true_features - pred_features))

# --- Custom Loss ---
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    perc_loss = perceptual_loss(y_true, y_pred)
    return 0.8 * mse + 0.2 * perc_loss

# --- Deeper U-Net Model with Two Inputs ---
def deeper_unet(input_shape=(128, 128, 1)):
    # Two inputs: current and PDN density
    input_current = Input(shape=input_shape, name="current_input")
    input_pdn = Input(shape=input_shape, name="pdn_input")
    input_dist = Input(shape=input_shape, name="dist_input")

    # Combine inputs
    merged_inputs = Concatenate()([input_current, input_pdn, input_dist])

    # Encoder
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(merged_inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.3)(c3)
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.4)(c4)
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.5)(c5)
    c5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.4)(c6)
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[input_current, input_pdn, input_dist], outputs=outputs)
    return model

# --- Compile and Train ---
model = deeper_unet(input_shape=(128, 128, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008), loss=custom_loss, metrics=["mae", "mse"])

train_ds = tf.data.Dataset.from_tensor_slices(((train_currents, train_pdns, train_dists), train_outputs)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(((test_currents, test_pdns, test_dists), test_outputs)).batch(32)

debug("Training the model...")
history = model.fit(train_ds, validation_data=test_ds, epochs=2000)

# --- Visualization ---
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss with Perceptual Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Generate predictions
debug("Generating predictions...")
predictions = model.predict(test_ds)

# Visualize inputs, ground truths, predictions, and calculate MAE
for i in range(10):
    # Calculate MAE for the current image
    mae = np.mean(np.abs(test_outputs[i].squeeze() - predictions[i].squeeze()))

    # Plot images
    plt.figure(figsize=(15, 5))  # Adjust figure size for more plots
    plt.subplot(1, 5, 1)  # 1 row, 5 columns
    plt.title("Input Current")
    plt.imshow(test_currents[i].squeeze(), cmap="viridis")

    plt.subplot(1, 5, 2)
    plt.title("Input PDN")
    plt.imshow(test_pdns[i].squeeze(), cmap="viridis")

    plt.subplot(1, 5, 3)
    plt.title("Input Effective Distance")
    plt.imshow(test_dists[i].squeeze(), cmap="viridis")

    plt.subplot(1, 5, 4)
    plt.title("Ground Truth")
    plt.imshow(test_outputs[i].squeeze(), cmap="viridis")

    plt.subplot(1, 5, 5)
    plt.title(f"Prediction\n(MAE: {mae:.4f})")  # Show MAE in the title
    plt.imshow(predictions[i].squeeze(), cmap="viridis")

    plt.show()
