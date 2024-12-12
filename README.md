# Static IR Drop Predictor

## A.1 Abstract
This artifact evaluation describes a comprehensive framework for static IR drop prediction in Power Delivery Networks (PDNs) using advanced U-Net-based deep learning models. The artifact includes an end-to-end implementation featuring code, datasets, and detailed instructions to replicate and extend the experiments conducted in the study. The methodology leverages a multi-input U-Net architecture to integrate spatial, topological, and electrical characteristics inherent to PDNs, offering robust and scalable solutions for static IR drop analysis. 

The artifact was evaluated using synthetic and real-world datasets derived from ICCAD 2023 Contest Problem C. Key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and runtime efficiency were used to validate model performance. The setup is hosted on Google Colab, employing NVIDIA T4 and L4 GPUs for efficient training and inference. This artifact enables users to explore model behavior, assess scalability, and extend the framework for broader applications in PDN analysis.

---

## A.2 Artifact Check-list (Meta-information)
- **Program:** Python implementation using TensorFlow and Keras frameworks.
- **Model:** U-Net-based architecture with multi-input handling, incorporating dense connections and hybrid loss functions.
- **Datasets:** ICCAD 2023 Contest Problem C datasets, including current maps, PDN density maps, effective distance maps, and IR drop maps.
- **OS Environment:** Ubuntu 20.04 with CUDA Toolkit 12.6 and cuDNN.
- **Hardware Setup:** Evaluated on Google Colab with NVIDIA T4 and L4 GPUs; preprocessing tasks conducted on Intel i7-class CPUs.
- **Execution:** Standalone Python scripts with GPU acceleration enabled.
- **Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Perceptual Loss, and runtime performance.
- **Output Format:** Training logs, evaluation metrics, graphical visualizations, and model checkpoints.
- **Disk Space:** Requires approximately 15 GB for datasets, intermediate files, and logs.
- **Setup Time:** Around 2 hours to prepare the environment, datasets, and dependencies.
- **Experiment Time:** Full training and evaluation completed in approximately 6 hours.
- **Availability:** Publicly accessible through GitHub.

---

## A.3 Description

### A.3.1 How to Access
The artifact is hosted on a public GitHub repository. Users can clone the repository and follow the `README.md` file for setup and execution instructions. The repository includes scripts for preprocessing, training, evaluation, and visualization, ensuring a seamless replication of the study.

### A.3.2 Hardware Dependencies
The experiments were conducted using Google Colab with NVIDIA T4 and L4 GPUs for efficient training and inference. The T4 GPU was utilized for U-Net training, while the L4 GPU was employed for Dense U-Net and GAN-based models due to their higher computational requirements. Preprocessing tasks were carried out using Intel i7-class CPUs for robust data handling. Other CUDA-compatible GPUs can also be used, with runtime variations depending on hardware specifications.

### A.3.3 Software Dependencies
The artifact uses Python as the primary programming language with libraries including TensorFlow (v2.8), Keras, NumPy, pandas, Matplotlib, and TensorFlow-Addons for advanced loss functions. CUDA 12.6 and cuDNN are required for GPU acceleration. A `requirements.txt` file is provided to facilitate dependency installation, ensuring compatibility across different environments.

### A.3.4 Datasets
The ICCAD 2023 Contest Problem C datasets were utilized, containing high-resolution synthetic and real-world circuits. Each sample comprises current maps, PDN density maps, effective distance maps, and corresponding IR drop maps. These datasets were preprocessed to ensure consistency in input shapes and value ranges, with training and testing subsets created to validate model performance comprehensively.

### A.3.5 Models
The artifact encompasses three primary models—U-Net, Dense U-Net, and GAN-based architectures—each tailored for specific aspects of static IR drop prediction in Power Delivery Networks (PDNs). These models leverage deep learning techniques to capture the intricate spatial, topological, and electrical characteristics inherent in PDNs.

1. **U-Net Architecture:**  
   The U-Net model serves as the foundational architecture, featuring a symmetric encoder-decoder design. The encoder extracts hierarchical features using convolutional layers and max-pooling, while the decoder reconstructs spatial resolutions through upsampling. Skip connections bridge encoder and decoder levels, retaining fine-grained details critical for precise IR drop prediction. It was trained on an NVIDIA T4 GPU with a learning rate of 0.0001, a batch size of 16, and 500 epochs.

2. **Dense U-Net Architecture:**  
   The Dense U-Net extends the U-Net by incorporating dense connectivity patterns within each block, enhancing feature reuse and improving gradient flow. This model captures complex spatial and topological variations, particularly in high-density PDNs. Dense U-Net integrates attention mechanisms for critical regions prone to significant IR drops and was trained on an NVIDIA L4 GPU for efficient runtime performance.

3. **GAN-Based Architecture:**  
   The Generative Adversarial Network (GAN) framework comprises a Dense U-Net-based generator and a discriminator. The generator predicts IR drop distributions, while the discriminator evaluates the realism of predictions. GAN excels in high-gradient regions, offering sharper and more realistic outputs. It was trained on an NVIDIA L4 GPU to handle intensive adversarial training.

---

## A.4 Installation
1. Clone the repository from GitHub.
2. Install the required Python libraries using `pip install -r requirements.txt`.
3. Verify the setup by running a sample test script included in the repository.
4. If using Google Colab, upload the repository and datasets to Colab, ensuring GPU acceleration is enabled in runtime settings.

---

## A.5 Experiment Workflow
1. **Data Preprocessing:** Normalize and resize datasets to $128 \times 128$ using the provided preprocessing scripts.  
2. **Training:** Execute the training script with a learning rate of 0.0001, batch size of 16, and 500 epochs.  
3. **Evaluation:** Evaluate the trained model on the test dataset using metrics such as MAE and MSE.  
4. **Visualization:** Generate plots comparing predicted vs. actual IR drops and training vs. validation loss curves.  
5. **Customization:** Modify scripts to explore alternative architectures, datasets, or loss functions.

---

## A.6 Evaluation and Expected Results
- **Training Logs:** Captures training and validation losses over epochs.  
- **Testing Metrics:** MAE, MSE, and runtime metrics for test datasets.  
- **Visualizations:** Side-by-side comparisons of predicted and actual IR drops with loss curves.  
- **Model Checkpoints:** Saved weights of trained models for further exploration.

---

## A.7 Experiment Customization
Users can:
- Adjust hyperparameters directly in the training scripts.
- Replace default datasets with new ones by updating file paths.
- Integrate custom loss functions by modifying `losses.py`.
- Experiment with alternative architectures by creating new models in the `models` directory.

---

## A.8 Methodology
The methodology ensures reproducibility and scalability for static IR drop prediction. Preprocessing involved normalizing feature maps to a range of 0 to 1 and resizing to $128 \times 128$. A 90:10 train-test split ensured robust evaluation. Training used U-Net, Dense U-Net, and GAN architectures with a hybrid loss function combining MSE and perceptual loss for structural fidelity. Training was conducted on Google Colab with NVIDIA T4 and L4 GPUs using a learning rate of 0.0001, batch size of 16, and 500 epochs. Evaluation metrics included MAE, MSE, and visual comparisons of predicted vs. actual IR drops. Robustness testing confirmed performance across varied dataset complexities and hardware configurations, establishing a scalable and efficient framework for static IR drop prediction.
