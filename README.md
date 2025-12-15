## SeeMyDigit- Handwritten Digit Recognition

**Project Overview**

SeeMyDigit is a classic Deep Learning and Computer Vision project that focuses on identifying digits (0–9) from handwritten images. This project uses two different Deep Learning models trained on the MNIST dataset to analyze the effect of image resolution on model performance.

I compare:
1. A model trained on original MNIST images (28×28)

2. A model trained on resized images (128×128)

This comparison helps understand how image resizing impacts accuracy.

**Objectives**

1. Implement handwritten digit recognition using Deep Learning
2. Compare model performance using different image sizes
3. Analyze accuracy, loss, and training efficiency
4. Understand the impact of higher resolution images on CNN models

**Technologies Used**

Programming Language: Python
Frameworks: Pytorch
Libraries: NumPy, Matplotlib, OpenCV and Scikit-learn etc.
Dataset: MNIST Handwritten Digit Dataset

**Dataset Description**

Dataset Name: MNIST
Total Images: 70,000
Training: 60,000
Testing: 10,000
Image Size: 28 × 28 pixels
Number of Classes: 10 (digits 0–9)

**System Architecture**

Step1: Input handwritten digit image
Step2: Image preprocessing (normalization & resizing)
Step3: Deep Learning model (CNN)
Step4: Feature extraction using convolutional layers
Step5: Fully connected layers
Step6: Output digit prediction

**Model Implementation**

1. Model 1: Original MNIST Image Size
   Input Size: 28 × 28
   Preprocessing: Normalization only
   Epochs: 6
   Accuracy: ~97–98%

   <img width="636" height="306" alt="image" src="https://github.com/user-attachments/assets/89eab16e-ba5b-4f4c-b3f8-f0073e13a28f" />


3. Model 2: Resized MNIST Image
   Input Size: 128 × 128
   Preprocessing: Resizing + normalization
   Epochs: 10
   Accuracy: slightly improved.

   <img width="653" height="530" alt="image" src="https://github.com/user-attachments/assets/8f493544-fc80-4bc3-b2a2-80fc40e0eed1" />

**Training Details**

Loss Function: Categorical Cross-Entropy
Optimizer: Adam
Activation Functions: ReLU
Evaluation Metrics: Accuracy, Loss

**Best Overall Choice: 128×128 Model**

For handwritten digit recognition using the MNIST dataset, the 128×128 input model is superior in terms of Accuracy.





