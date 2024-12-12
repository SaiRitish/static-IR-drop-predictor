# static-IR-drop-predictor

A.1 Abstract
This artifact evaluation describes a comprehensive frame- work for static IR drop prediction in Power Delivery Networks (PDNs) using advanced U-Net-based deep learning models. The artifact includes an end-to-end implementation featuring code, datasets, and detailed instructions to replicate and extend the experiments conducted in the study. The methodology leverages a multi-input U-Net architecture to integrate spatial, topological, and electrical characteristics inherent to PDNs, offering robust and scalable solutions for static IR drop analy- sis. The artifact was evaluated using synthetic and real-world datasets derived from ICCAD 2023 Contest Problem C. Key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), runtime efficiency were used to validate model performance. The setup is hosted on Google Colab, employing NVIDIA T4 and L4 GPUs for efficient training and inference. This artifact enables users to explore model behavior, assess scalability, and extend the framework for broader applications in PDN analysis.
A.2 Artifact Check-list (Meta-information)
• Program: Python implementation using TensorFlow and Keras frameworks.
• Model: U-Net-based architecture with multi-input han- dling, incorporating dense connections and hybrid loss functions.
• Datasets: ICCAD 2023 Contest Problem C datasets, including current maps, PDN density maps, effective distance maps, and IR drop maps.
• OS Environment: Ubuntu 20.04 with CUDA Toolkit 12.6 and cuDNN.
• Hardware Setup: Evaluated on Google Colab with NVIDIA T4 and L4 GPUs; preprocessing tasks conducted on Intel i7-class CPUs.
• Execution: Standalone Python scripts with GPU acceler- ation enabled.
• Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Perceptual Loss, and runtime performance.
• Output Format: Training logs, evaluation metrics,
graphical visualizations, and model checkpoints.
• Disk Space: Requires approximately 15 GB for datasets,
intermediate files, and logs.
• Setup Time: Around 2 hours to prepare the environment,
datasets, and dependencies.
• Experiment Time: Full training and evaluation com-
pleted in approximately 6 hours.
• Availability: Publicly accessible through GitHub.
A.3 Description
A.3.1 How to Access: The artifact is hosted on a public GitHub repository. Users can clone the repository and follow the README file for setup and execution instructions. The
repository includes scripts for preprocessing, training, evalua- tion, and visualization, ensuring a seamless replication of the study.
A.3.2 Hardware Dependencies: The experiments were con- ducted using Google Colab with NVIDIA T4 and L4 GPUs for efficient training and inference. The T4 GPU was utilized for U-Net training, while the L4 GPU was employed for Dense U-Net and GAN-based models due to their higher computational requirements. Preprocessing tasks were carried out using Intel i7-class CPUs for robust data handling. Other CUDA-compatible GPUs can also be used, with runtime variations depending on hardware specifications.
A.3.3 Software Dependencies: The artifact uses Python as the primary programming language with libraries includ- ing TensorFlow (v2.8), Keras, NumPy, pandas, Matplotlib, and TensorFlow-Addons for advanced loss functions. CUDA 12.6 and cuDNN are required for GPU acceleration. A requirements.txt file is provided to facilitate depen- dency installation, ensuring compatibility across different en- vironments.
A.3.4 Datasets: The ICCAD 2023 Contest Problem C datasets were utilized, containing high-resolution synthetic and real-world circuits. Each sample comprises current maps, PDN density maps, effective distance maps, and corresponding IR drop maps. These datasets were preprocessed to ensure consistency in input shapes and value ranges, with training and testing subsets created to validate model performance comprehensively.
A.3.5 Models: The artifact encompasses three primary models—U-Net, Dense U-Net, and GAN-based architec- tures—each tailored for specific aspects of static IR drop prediction in Power Delivery Networks (PDNs). These models leverage deep learning techniques to capture the intricate spatial, topological, and electrical characteristics inherent in PDNs.
a) 1. U-Net Architecture: The U-Net model serves as the foundational architecture, featuring a symmetric encoder- decoder design. The encoder extracts hierarchical features from the input data using a series of convolutional layers and max-pooling operations, while the decoder reconstructs spatial resolutions through upsampling layers. Skip connec- tions bridge the encoder and decoder at corresponding levels, ensuring the retention of fine-grained spatial details critical for precise IR drop prediction. The U-Net model is optimized for computational efficiency and is particularly effective for low- complexity PDNs. It was trained on an NVIDIA T4 GPU, utilizing a learning rate of 0.0001, a batch size of 16, and 500 epochs.
b) 2. Dense U-Net Architecture: The Dense U-Net ex- tends the traditional U-Net by incorporating dense connectivity patterns within each block. In this architecture, each layer receives input from all preceding layers in the same block, fostering feature reuse and improving gradient flow. This dense connectivity enhances the model’s ability to capture complex spatial and topological variations, particularly in high-density PDNs. The Dense U-Net also integrates attention
mechanisms to emphasize critical regions prone to significant IR drops, boosting its structural accuracy. Training the Dense U-Net required the computational power of an NVIDIA L4 GPU, which provided the necessary resources for handling the model’s increased complexity while maintaining efficient runtime performance.
c) 3. GAN-Based Architecture: The Generative Adver- sarial Network (GAN) architecture comprises a generator and a discriminator, working in an adversarial framework to produce high-fidelity predictions. The generator is modeled as a Dense U-Net, tasked with predicting IR drop distributions, while the discriminator evaluates the realism of the generated predictions. The GAN framework promotes sharper and more realistic outputs by balancing the pixel-wise accuracy of the generator with the structural consistency enforced by the discriminator. This model is particularly effective in high- gradient regions, where traditional approaches may struggle with oversmoothing. The GAN-based architecture was trained using an NVIDIA L4 GPU to accommodate the intensive adversarial training process.
A.4 Installation
To set up the artifact:
1) Clone the repository from GitHub.
2) Install the required Python libraries using pip
install -r requirements.txt.
3) Verify the setup by running a sample test script included
in the repository.
4) If using Google Colab, upload the repository and
datasets to Colab, ensuring GPU acceleration is enabled in runtime settings.
A.5 Experiment Workflow
The artifact follows a structured workflow to simplify ex- perimentation:
• Data Preprocessing: Normalize and resize the datasets using the provided preprocessing scripts. The inputs are aligned to a consistent shape (128×128) and normalized to a range of 0 to 1 for numerical stability.
• Training: Execute the training script with default hyper- parameters, including a learning rate of 0.0001, batch size of 16, and 500 epochs.
• Evaluation: Evaluate the trained model on the test dataset, calculating metrics such as MAE and MSE.
• Visualization: Generate plots comparing predicted and actual IR drop distributions, as well as training and validation loss curves.
• Customization: Modify the scripts to explore alternative architectures, datasets, or loss functions.
A.6 Evaluation and Expected Results
The artifact produces the following outputs:
• Training Logs: Detailed logs capturing training and validation losses over epochs.
• Testing Metrics: MAE, MSE, and runtime metrics for test datasets, offering insights into model performance.
• Visualizations: Side-by-side comparisons of predicted vs. actual IR drops, along with loss curves over epochs. • Model Checkpoints: Saved weights of trained models,
enabling further exploration and extension.
A.7 Experiment Customization
The codebase is designed for extensibility, allowing users to:
• Adjust hyperparameters directly in the training scripts (e.g., batch size, learning rate, and number of epochs). • Replace default datasets with new data by updating file
paths in the preprocessing scripts.
• Integrate custom loss functions by modifying the
losses.py module.
• Experiment with alternative architectures by creating new
models in the models directory. A.8 Methodology
The artifact employs a structured methodology to ensure reproducibility and scalability for static IR drop prediction. Datasets from ICCAD 2023 Contest Problem C, including synthetic and real-world circuits, were preprocessed by nor- malizing feature maps (current, PDN density, effective dis- tance, and IR drop maps) to a range of 0 to 1 and resizing them to 128 × 128 for consistency. A 90:10 train-test split ensured robust evaluation, while optional data augmentation techniques, such as rotations and flips, improved robustness.
Model training utilized U-Net, Dense U-Net, and GAN architectures with a hybrid loss function combining MSE for pixel-level accuracy and perceptual loss for preserving struc- tural details using VGG19 features. Training was conducted on Google Colab using NVIDIA T4 GPUs for U-Net and L4 GPUs for the more complex Dense U-Net and GAN models. Default settings included a learning rate of 0.0001, a batch size of 16, and 500 epochs, with optional early stopping to prevent overfitting.
Evaluation focused on accuracy and fidelity, using metrics like MAE and MSE, alongside visual comparisons of predicted and actual IR drop maps. Controlled environments ensured reproducibility, and pre-trained model checkpoints were saved for further use. Robustness was tested across varying dataset complexities and hardware configurations. The models demon- strated adaptability on diverse GPUs (T4 and L4) and CPUs, with stress tests confirming reliable performance in challeng- ing regions such as high-gradient zones.
This streamlined framework establishes a scalable and efficient approach for static IR drop prediction, validating the models’ performance and offering a solid foundation for advancements in PDN analysis.
