# Anomaly Detection in Diffusion MRI for Brain Tumor Patients
## Project

This work focuses on anomaly detection in diffusion MRI for brain tumor patients. For this purpose, 
two different deep learning approaches were developed and optimized, which were exclusively trained on non-annotated 
brain images of healthy subjects. The first approach is a proof-of-concept denoising autoencoder, to demonstrate the 
general feasibility. Moreover, a more sophisticated model consisting of a reconstruction- and a discrimination network 
was developed. To validate them, a state-of-the-art supervised tumor segmentation network was modified to generate a 
ground truth for the available tumor patients. For this, existing MRI data were segmented and then registered to the 
diffusion images. The approaches clearly show that diffusion images are suitable for tumor segmentation by using 
anomaly detection methods. Good results were obtained both qualitatively and quantitatively. With an unsupervised 
threshold, a Dice score of 0.62 +- 0.2 was achieved. By using the combined advantage of post-processing and 
an optimized threshold, the score could be improved to 0.67 +- 0.2. This work demonstrates the fundamental 
potential of anomaly detection in diffusion MRI and provides a basis for further optimization.

For further information about the topic please feel free to write me an email: jarek.ecke@rwth-aachen.de

## Instruction

DataProcessing
- groundtruth.py
In order to evaluate the anomaly detection, a ground truth must be created. Using the script, a U-Net is 
trained on the BraTS 2021 data and then tested on the subjects from UKA. The script guides through the entire process 
to the final masks on the MRI data for this project. Registration to dMRI takes place externally. Access to the 
UKA data and the BraTS 2021 data is a prerequisite.

- processing_data.py
Herewith the UKA data can be preprocessed for this project. Furthermore, all folder structures are created correctly. 
Access to all dMRI datasets and the corresponding masks for the tumors are required.

- qualitative_analysis.py and quantitative_analysis.py
These scripts access the results of the trained networks and can be used to evaluate them.

main.py
- If all images are available and completely preprocessed, the networks can be trained or tested using main.py. All 
possible settings can be looked up in config.py and passed as arguments.

Models
- Models includes all deep learning models used here. The individual networks can be modified. 

## Data
 
- UKA dataset (in-house)
    - 28 diffusion MRI images with 64 channels for training
    - 32 diffusion MRI images with 64 channels for testing
- BraTS 2021 data (public)
    - 1251 training datasets from BraTS 2021 with T1, T1ce, T2 and FLAIR volumes

## Requirements

- GPU with 24GB VRAM and CUDA (for most settings 12GB VRAM are sufficient)
- numpy 1.21.5
- pytorch-lightning 1.21.5
- torch 1.11.0
- scikit-learn 1.0.2
- matplotlib 3.3.2