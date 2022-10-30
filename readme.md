# Anomaly Detection in Diffusion MRI for Brain Tumor Patients
Project

This project focuses on anomaly detection in diffusion MRI for brain tumor patients. For this purpose, 
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

Installation 




Instruction