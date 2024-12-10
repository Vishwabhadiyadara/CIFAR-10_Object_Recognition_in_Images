# CIFAR-10_Object_Recognition_in_Images
# Table of Contents
1. [Abstract](#abstract)
2. [Overview](#overview)
   - [What is the problem?](#what-is-the-problem)
   - [Why is this problem interesting?](#why-is-this-problem-interesting)
   - [Use cases](#use-cases)
3. [Approach](#approach)
   - [Proposed Methodology](#proposed-methodology)
   - [Rationale](#rationale)
   - [Key Components and Results](#key-components-and-results)
4. [Experiment Setup](#experiment-setup)
   - [Dataset](#dataset)
   - [Implementation Details](#implementation-details)
   - [Computing Environment](#computing-environment)
5. [Experiment Results](#experiment-results)
   - [Main Results](#main-results)
   - [Supplementary Results](#supplementary-results)
6. [Discussion](#discussion)
   - [Comparative Analysis](#comparative-analysis)
   - [Challenges](#challenges)
7. [Conclusion](#conclusion)
8. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)

## Team Members
Vishwa Bhadiyadara

Bhakti Pasnani

Vaishnvi Bhutada

## Abstract
This project aims to classify images from the CIFAR-100 dataset into 100 distinct categories using machine learning models. The models explored include Logistic Regression, Logistic Regression with PCA, and Convolutional Neural Networks (CNNs). The CNN-based approach demonstrates superior performance, showcasing the potential of deep learning for image classification tasks.

## Overview
The CIFAR-100 dataset contains 60,000 color images of size 32x32 pixels, categorized into 100 classes. This project explores various approaches to classify these images effectively:
- Logistic Regression: A baseline approach to evaluate the dataset's linear separability.
- Logistic Regression with PCA: Dimensionality reduction to improve computational efficiency.
- Convolutional Neural Networks (CNNs): A state-of-the-art deep learning approach leveraging data augmentation and advanced architectures.

The project demonstrates significant improvements in classification accuracy with the CNN model, which outperforms traditional methods.

---

## Problem Statement
### What is the problem?
The goal is to classify images from the CIFAR-100 dataset into their respective categories. This involves recognizing diverse objects, animals, and scenes under varying conditions.

### Why is this problem interesting?
Image classification is fundamental in many AI applications, including autonomous vehicles, medical imaging, and surveillance systems. Successfully solving this problem contributes to broader advancements in computer vision.

### Use cases:
- Educational tools for visual recognition.
- Enhancing search engines with image-based retrieval.
- Automated categorization in e-commerce platforms.

---

## Approach
### Proposed Methodology
1. **Baseline Models**: Logistic Regression to establish foundational metrics.
2. **Dimensionality Reduction**: PCA to reduce data complexity and improve performance.
3. **Deep Learning**: A CNN architecture with data augmentation and hyperparameter optimization.

### Rationale
The choice of CNNs is informed by their proven ability to capture spatial hierarchies in image data. Compared to traditional methods, CNNs better exploit the dataset's features, yielding higher accuracy.

### Key Components and Results
1. **Baseline Accuracy**:
   - Logistic Regression: 17.98%
   - Logistic Regression with PCA: 18.61%
2. **CNN Accuracy**: 37.74%

**Limitations**:
- High computational requirements for training CNNs.
- Moderate accuracy, indicating potential for further optimization.

---

## Experiment Setup
### Dataset
- **Source**: CIFAR-100
- **Statistics**: 60,000 images, 100 classes, 500 training and 100 test images per class.

### Implementation Details
- **Baseline Models**: Implemented using scikit-learn.
- **CNN Architecture**:
  - Convolutional layers with Batch Normalization and Dropout.
  - Optimizer: Adam with learning rate 0.001.
  - Loss Function: Categorical Crossentropy.
- **Data Augmentation**: Rotation, width/height shift, horizontal flip, and zoom.

### Computing Environment
- Framework: TensorFlow/Keras
- Accelerator: NVIDIA T4 GPU (Google Colab)

---

## Experiment Results
### Main Results
- CNN achieves the highest accuracy (37.74%) compared to Logistic Regression models.
- Data augmentation significantly boosts model robustness.

### Supplementary Results
- Optimized hyperparameters for CNN layers and dropout rates.
- Evaluated the impact of dimensionality reduction on Logistic Regression.

---

## Discussion
### Comparative Analysis
The CNN model shows clear advantages in capturing complex patterns compared to traditional models. However, the accuracy suggests room for improvement, such as:
- Exploring deeper networks or transfer learning.
- Fine-tuning data augmentation strategies.

### Challenges
- Computational overhead for CNN training.
- Moderate accuracy relative to state-of-the-art benchmarks.

---

## Conclusion
This project effectively demonstrates the application of machine learning and deep learning techniques to the CIFAR-100 image classification task, a challenging dataset with 100 diverse categories. Through comparative analysis of Logistic Regression, Logistic Regression with PCA, and a Convolutional Neural Network (CNN), the study underscores the clear advantages of deep learning for such complex tasks. The CNN approach, bolstered by data augmentation and optimized architecture, achieved a significant improvement in accuracy (37.74%) compared to traditional models, showcasing its capability to capture intricate patterns in image data.

While the CNN outperformed other approaches, its accuracy highlights room for further optimization. Future work could explore deeper architectures, hyperparameter tuning, and transfer learning to enhance performance. This project not only provides a strong foundation for advancing image classification models but also illustrates the importance of leveraging modern deep learning techniques to tackle real-world challenges in computer vision. The findings are a step forward in developing robust solutions for diverse applications, from autonomous systems to automated content tagging.

---

## Getting Started
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CIFAR-100-Classification.git
   cd CIFAR-100-Classification
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

### Usage
1. Train the model:
   ```bash
   python train_model.py

2. Test the model:
   ```bash
   python test_model.py

3. Predict on custom images:
   ```bash
   python predict_image.py --image path/to/image.jpg



