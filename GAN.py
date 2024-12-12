import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate,
    BatchNormalization, LeakyReLU, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision for better performance
set_global_policy('mixed_float16')

# --- Debugging Helper ---
def debug(message):
    print(f"[DEBUG] {message}")

# --- File Paths ---
base_path = "/content/drive/My Drive/ML-for-IR-drop-main/benchmarks/fake-circuit-data/"

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

# Sort files by index
def extract_index(file_name):
    return int(file_name.split("map")[1].split("_")[0])

available_current_files.sort(key=lambda f: extract_index(f))
available_pdn_files.sort(key=lambda f: extract_index(f))
available_dist_files.sort(key=lambda f: extract_index(f))
available_ir_drop_files.sort(key=lambda f: extract_index(f))

debug(f"Found {len(available_current_files)} current files.")

# --- Preprocessing ---
def preprocess_csv(file_path, target_shape=(128, 128)):
    debug(f"Processing: {file_path}")
    data = pd.read_csv(file_path, header=None).to_numpy(dtype=np.float32)
    data = tf.image.resize(data[..., np.newaxis], target_shape).numpy()
    max_val = np.max(data)
    return data / max_val if max_val > 0 else data

# Load and preprocess data
input_currents = np.array([preprocess_csv(file) for file in available_current_files])
input_pdns = np.array([preprocess_csv(file) for file in available_pdn_files])
input_dists = np.array([preprocess_csv(file) for file in available_dist_files])
output_images = np.array([preprocess_csv(file) for file in available_ir_drop_files])

# Split data
split_idx = len(input_currents) - 10
train_currents, test_currents = input_currents[:split_idx], input_currents[split_idx:]
train_pdns, test_pdns = input_pdns[:split_idx], input_pdns[split_idx:]
train_dists, test_dists = input_dists[:split_idx], input_dists[split_idx:]
train_outputs, test_outputs = output_images[:split_idx], output_images[split_idx:]

debug(f"Training samples: {len(train_currents)}, Testing samples: {len(test_currents)}")

# --- Generator (Dense UNet) ---
def dense_unet(input_shape=(128, 128, 1)):
    def dense_block(x, filters, num_layers):
        for _ in range(num_layers):
            bn = BatchNormalization()(x)
            act = tf.keras.activations.relu(bn)
            conv = Conv2D(filters, (3, 3), padding="same")(act)
            x = Concatenate()([x, conv])
        return x

    def transition_down(x, filters):
        x = Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    def transition_up(x, filters):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        return x

    input_current = Input(shape=input_shape, name="current_input")
    input_pdn = Input(shape=input_shape, name="pdn_input")
    input_dist = Input(shape=input_shape, name="dist_input")
    merged_inputs = Concatenate()([input_current, input_pdn, input_dist])

    d1 = dense_block(merged_inputs, 64, 4)
    t1 = transition_down(d1, 128)
    d2 = dense_block(t1, 128, 4)
    t2 = transition_down(d2, 256)
    d3 = dense_block(t2, 256, 4)
    t3 = transition_down(d3, 512)
    d4 = dense_block(t3, 512, 4)
    t4 = transition_down(d4, 1024)

    bottleneck = dense_block(t4, 1024, 4)

    u1 = transition_up(bottleneck, 512)
    u1 = Concatenate()([u1, d4])
    u1 = dense_block(u1, 512, 4)
    u2 = transition_up(u1, 256)
    u2 = Concatenate()([u2, d3])
    u2 = dense_block(u2, 256, 4)
    u3 = transition_up(u2, 128)
    u3 = Concatenate()([u3, d2])
    u3 = dense_block(u3, 128, 4)
    u4 = transition_up(u3, 64)
    u4 = Concatenate()([u4, d1])
    u4 = dense_block(u4, 64, 4)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u4)
    model = Model(inputs=[input_current, input_pdn, input_dist], outputs=outputs)
    return model

# --- Discriminator (PatchGAN) ---
def build_discriminator(input_shape=(128, 128, 1)):
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, (4, 4), strides=2, padding="same")(input_layer)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (4, 4), padding="same")(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

# --- Loss Functions ---
def generator_loss(disc_generated_output, gen_output, target):
    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output
    )
    l1_loss = tf.keras.losses.MeanAbsoluteError()(target, gen_output)
    return adv_loss + 100 * l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output
    )
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    return real_loss + fake_loss

# --- Training Step ---
@tf.function
def train_step(input_data, target, generator, discriminator, gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_data, training=True)
        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# --- Training Loop ---
def train_gan(generator, discriminator, dataset, epochs, gen_optimizer, disc_optimizer):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for input_data, target in dataset:
            gen_loss, disc_loss = train_step(input_data, target, generator, discriminator, gen_optimizer, disc_optimizer)
        print(f"Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")

# Initialize models and optimizers
generator = dense_unet()
discriminator = build_discriminator()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Prepare dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    ((train_currents, train_pdns, train_dists), train_outputs)
).batch(8).prefetch(tf.data.experimental.AUTOTUNE)

# Train GAN
train_gan(generator, discriminator, train_ds, epochs=500, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)

# --- Visualization and Prediction ---
def visualize_predictions(generator, input_currents, input_pdns, input_dists, target):
    predictions = generator([input_currents, input_pdns, input_dists], training=False)
    num_samples = min(5, len(input_currents))  # Visualize up to 5 samples
    plt.figure(figsize=(20, 10))

    for i in range(num_samples):
        # Calculate MSE and MAE
        mse = tf.keras.losses.MeanSquaredError()(target[i], predictions[i]).numpy()
        mae = tf.keras.losses.MeanAbsoluteError()(target[i], predictions[i]).numpy()
        print(f"Sample {i+1}: MSE = {mse:.4f}, MAE = {mae:.4f}")

        # Current Input
        plt.subplot(num_samples, 5, i * 5 + 1)
        plt.title("Current Input")
        plt.imshow(input_currents[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # PDN Input
        plt.subplot(num_samples, 5, i * 5 + 2)
        plt.title("PDN Input")
        plt.imshow(input_pdns[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Distance Input
        plt.subplot(num_samples, 5, i * 5 + 3)
        plt.title("Distance Input")
        plt.imshow(input_dists[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Ground Truth
        plt.subplot(num_samples, 5, i * 5 + 4)
        plt.title("Ground Truth")
        plt.imshow(target[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Prediction
        plt.subplot(num_samples, 5, i * 5 + 5)
        plt.title(f"Prediction\nMSE: {mse:.4f}, MAE: {mae:.4f}")
        plt.imshow(predictions[i, ..., 0], cmap="viridis")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run visualization on test data
visualize_predictions(generator, test_currents, test_pdns, test_dists, test_outputs)


########################################################################################################################################################################################################################################################################

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate,
    BatchNormalization, LeakyReLU, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision for better performance
set_global_policy('mixed_float16')

# --- Debugging Helper ---
def debug(message):
    print(f"[DEBUG] {message}")

# --- File Paths ---
base_path = "/content/drive/My Drive/ML-for-IR-drop-main/benchmarks/fake-circuit-data/"

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

# Sort files by index
def extract_index(file_name):
    return int(file_name.split("map")[1].split("_")[0])

available_current_files.sort(key=lambda f: extract_index(f))
available_pdn_files.sort(key=lambda f: extract_index(f))
available_dist_files.sort(key=lambda f: extract_index(f))
available_ir_drop_files.sort(key=lambda f: extract_index(f))

debug(f"Found {len(available_current_files)} current files.")

# --- Preprocessing ---
def preprocess_csv(file_path, target_shape=(128, 128)):
    debug(f"Processing: {file_path}")
    data = pd.read_csv(file_path, header=None).to_numpy(dtype=np.float32)
    data = tf.image.resize(data[..., np.newaxis], target_shape).numpy()
    max_val = np.max(data)
    return data / max_val if max_val > 0 else data

# Load and preprocess data
input_currents = np.array([preprocess_csv(file) for file in available_current_files])
input_pdns = np.array([preprocess_csv(file) for file in available_pdn_files])
input_dists = np.array([preprocess_csv(file) for file in available_dist_files])
output_images = np.array([preprocess_csv(file) for file in available_ir_drop_files])

# Split data
split_idx = len(input_currents) - 10
train_currents, test_currents = input_currents[:split_idx], input_currents[split_idx:]
train_pdns, test_pdns = input_pdns[:split_idx], input_pdns[split_idx:]
train_dists, test_dists = input_dists[:split_idx], input_dists[split_idx:]
train_outputs, test_outputs = output_images[:split_idx], output_images[split_idx:]

debug(f"Training samples: {len(train_currents)}, Testing samples: {len(test_currents)}")

# --- Generator (Dense UNet) ---
def dense_unet(input_shape=(128, 128, 1)):
    def dense_block(x, filters, num_layers):
        for _ in range(num_layers):
            bn = BatchNormalization()(x)
            act = tf.keras.activations.relu(bn)
            conv = Conv2D(filters, (3, 3), padding="same")(act)
            x = Concatenate()([x, conv])
        return x

    def transition_down(x, filters):
        x = Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    def transition_up(x, filters):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        return x

    input_current = Input(shape=input_shape, name="current_input")
    input_pdn = Input(shape=input_shape, name="pdn_input")
    input_dist = Input(shape=input_shape, name="dist_input")
    merged_inputs = Concatenate()([input_current, input_pdn, input_dist])

    d1 = dense_block(merged_inputs, 64, 4)
    t1 = transition_down(d1, 128)
    d2 = dense_block(t1, 128, 4)
    t2 = transition_down(d2, 256)
    d3 = dense_block(t2, 256, 4)
    t3 = transition_down(d3, 512)
    d4 = dense_block(t3, 512, 4)
    t4 = transition_down(d4, 1024)

    bottleneck = dense_block(t4, 1024, 4)

    u1 = transition_up(bottleneck, 512)
    u1 = Concatenate()([u1, d4])
    u1 = dense_block(u1, 512, 4)
    u2 = transition_up(u1, 256)
    u2 = Concatenate()([u2, d3])
    u2 = dense_block(u2, 256, 4)
    u3 = transition_up(u2, 128)
    u3 = Concatenate()([u3, d2])
    u3 = dense_block(u3, 128, 4)
    u4 = transition_up(u3, 64)
    u4 = Concatenate()([u4, d1])
    u4 = dense_block(u4, 64, 4)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u4)
    model = Model(inputs=[input_current, input_pdn, input_dist], outputs=outputs)
    return model

# --- Discriminator (PatchGAN) ---
def build_discriminator(input_shape=(128, 128, 1)):
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, (4, 4), strides=2, padding="same")(input_layer)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (4, 4), strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (4, 4), padding="same")(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

# --- Loss Functions ---
def generator_loss(disc_generated_output, gen_output, target):
    # Adversarial loss
    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_generated_output), disc_generated_output
    )
    # L1 loss for pixel-wise differences
    l1_loss = tf.keras.losses.MeanAbsoluteError()(target, gen_output)
    # SSIM loss for structural similarity
    ssim_loss = 1 - tf.reduce_mean(
        tf.image.ssim(
            tf.cast(gen_output, tf.float32),
            tf.cast(target, tf.float32),
            max_val=1.0
        )
    )
    # Combined loss
    return adv_loss + 100 * l1_loss + 10 * ssim_loss



def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output
    )
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    return real_loss + fake_loss

# --- Training Step ---
@tf.function
def train_step(input_data, target, generator, discriminator, gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_data, training=True)
        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# --- Training Loop ---
def train_gan(generator, discriminator, dataset, epochs, gen_optimizer, disc_optimizer):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for input_data, target in dataset:
            gen_loss, disc_loss = train_step(input_data, target, generator, discriminator, gen_optimizer, disc_optimizer)
        print(f"Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")

# Initialize models and optimizers
generator = dense_unet()
discriminator = build_discriminator()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Prepare dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    ((train_currents, train_pdns, train_dists), train_outputs)
).batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# Train GAN
train_gan(generator, discriminator, train_ds, epochs=500, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)

# --- Visualization and Prediction ---
def visualize_predictions(generator, input_currents, input_pdns, input_dists, target):
    predictions = generator([input_currents, input_pdns, input_dists], training=False)
    num_samples = min(5, len(input_currents))  # Visualize up to 5 samples
    plt.figure(figsize=(20, 10))

    for i in range(num_samples):
        mse = tf.keras.losses.MeanSquaredError()(target[i], predictions[i]).numpy()
        mae = tf.keras.losses.MeanAbsoluteError()(target[i], predictions[i]).numpy()

        # Current Input
        plt.subplot(num_samples, 5, i * 5 + 1)
        plt.title("Current Input")
        plt.imshow(input_currents[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # PDN Input
        plt.subplot(num_samples, 5, i * 5 + 2)
        plt.title("PDN Input")
        plt.imshow(input_pdns[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Distance Input
        plt.subplot(num_samples, 5, i * 5 + 3)
        plt.title("Distance Input")
        plt.imshow(input_dists[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Ground Truth
        plt.subplot(num_samples, 5, i * 5 + 4)
        plt.title("Ground Truth")
        plt.imshow(target[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Prediction with MAE and MSE
        plt.subplot(num_samples, 5, i * 5 + 5)
        plt.title(f"Prediction\nMAE: {mae:.4f}\nMSE: {mse:.4f}")
        plt.imshow(predictions[i, ..., 0], cmap="viridis")
        plt.axis("off")

        # Print MAE and MSE
        print(f"Sample {i + 1}: MAE = {mae:.4f}, MSE = {mse:.4f}")

    plt.tight_layout()
    plt.show()

# Run visualization on test data
visualize_predictions(generator, test_currents, test_pdns, test_dists, test_outputs)

