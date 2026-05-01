# AI Systems for Bone Fracture Detection in X-Rays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20Imaging-green)

## 📌 Project Overview
This project explores the use of Deep Learning to automate fracture detection in X-ray images across diverse anatomical regions (hand, leg, hip, and shoulder). 

The study highlights a **multimodal approach**, fusing traditional radiographic image data with tabular metadata (scan angle, body part, and hardware status) to improve diagnostic accuracy. We benchmarked custom "handcrafted" CNNs against state-of-the-art pre-trained models (ResNet, VGG, ViT).

## 🚀 Key Innovation: Multimodal Fusion
Unlike standard image-only classifiers, our **Advanced Model** incorporates:
*   **Image Branch:** 3-layer regularized CNN processing 224x224 grayscale X-rays.
*   **Metadata Branch:** Fully connected layers processing categorical features (Body Part, Scan Angle, presence of Hardware/Pins).
*   **Concatenation Layer:** Merges features from both branches to provide a context-aware diagnostic prediction.

## 📊 Dataset
*   **Source:** FracAtlas Dataset (Publicly available on Kaggle).
*   **Size:** 4,083 X-ray images.
*   **Class Distribution:** 3,366 non-fractured / 717 fractured (Handled via Class Weighting).
*   **Preprocessing:** BGR to Grayscale conversion, resizing, and aggressive data augmentation (Random flips, 90° rotations, Jittering, and Random Erasing).

## 🛠️ Architectures Evaluated
We implemented and compared two distinct categories of models:

| Category | Models | Framework |
| :--- | :--- | :--- |
| **Handcrafted** | Simple CNN, Deep CNN, Regularized CNN, Advanced Multimodal CNN | TensorFlow/Keras |
| **Pre-trained** | ResNet50, VGG16, Vision Transformer (ViT) | PyTorch |
| **Ensemble** | Soft-voting Ensemble (ResNet50 + ViT) | PyTorch |

## 📈 Performance Results
Our findings confirm that **Transfer Learning** significantly outperforms training from scratch, even when bridging the gap from natural images (ImageNet) to medical radiographs.

| Model | Accuracy | ROC-AUC | F1-Score |
| :--- | :--- | :--- | :--- |
| **Advanced Handcrafted (Multimodal)** | 78.82% | 0.7916 | 0.4986 |
| **ResNet50 (Fine-tuned)** | **89.53%** | 0.9313 | 0.7217 |
| **ViT (Fine-tuned)** | 87.56% | 0.9159 | 0.6833 |
| **Ensemble (ResNet50 + ViT)** | 89.03% | **0.9347** | **0.7287** |

**Key Insight:** The ensemble of ResNet50 and ViT yielded the best balance of sensitivity and specificity, proving that architectural diversity leads to synergistic gains in medical diagnostics.

## 📂 Repository Structure
*   `handcrafted_models.ipynb`: TensorFlow implementation of custom CNNs and Multimodal networks.
*   `pretrained_models.py`: PyTorch implementation of ResNet50, VGG16, and ViT fine-tuning.
*   `Report.pdd`: The full technical study.

## 👥 Contributors
*   **Naman Dhaval Desai**
*   **Noah Márquez Vara**
