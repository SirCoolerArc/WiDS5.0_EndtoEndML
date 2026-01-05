# 🌱 Plant Disease Detection | End-to-End ML Systems with Federated Averaging | WiDS 5.0

**Project Duration:** 3 Weeks (Midterm Submission)  
**Domain:** Computer Vision, Agriculture, Deep Learning  
**Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 📌 Project Overview
This project explores the progression from traditional Machine Learning to state-of-the-art Deep Learning techniques for classifying plant leaf diseases. Working with the **PlantVillage dataset** (38 classes, ~54k images), the goal was to build a robust model capable of assisting in early disease diagnosis for crops like Apple, Corn, Potato, and Tomato.

The project follows a rigorous 3-week timeline, moving from Exploratory Data Analysis (EDA) to Feature Engineering, and finally to Transfer Learning.

---

## 📂 Repository Structure
This repository contains the code for the first three weeks of the project, organized into three primary notebooks:

| File/Notebook | Description |
| :--- | :--- |
| `Week1_EDA.ipynb` | **Data Analysis:** Data loading, class distribution visualization, and image quality assessment. |
| `Week2_Classical_ML.ipynb` | **Baselines:** Feature extraction (flattening) and training traditional models (SVM, Random Forest). |
| `Week3_Deep_Learning.ipynb` | **CNNs & Transfer Learning:** Custom CNN implementation vs. MobileNetV2 Transfer Learning. |

---

## 🗓️ Weekly Progress & Methodology

### **Week 1: Data Understanding & EDA**
* **Objective:** Understand the data structure and quality.
* **Key Findings:**
    * The dataset contains **38 classes** (e.g., *Apple___Black_rot*, *Corn___Healthy*).
    * Images are RGB (color) and of varying resolutions (resized to 224x224 for modeling).
    * Class imbalance was observed, which would later impact standard ML models.

### **Week 2: The Limits of Classical Machine Learning**
* **Objective:** Establish a baseline using "Old School" ML algorithms.
* **Approach:** Images were flattened into 1D vectors and fed into various classifiers.
* **Models Tested:**
    1.  **Dummy Classifier:** ~10% accuracy (Random chance).
    2.  **Linear SVM (with PCA):** ~66% accuracy.
    3.  **Random Forest (Unbounded):** ~68.5% accuracy (Best Performer).
* **Analysis:**
    * While Random Forest performed best, it struggled with "feature-poor" classes.
    * **Critical Failure:** The model achieved an **F1-score of 0.00** for *Apple Cedar Rust* and *Potato Healthy*, proving that pixel-based feature extraction was insufficient for complex texture recognition.

### **Week 3: The Deep Learning Shift**
* **Objective:** Learn features automatically using Convolutional Neural Networks (CNNs).
* **Approach 1: Simple CNN (From Scratch)**
    * Built a custom 2-layer CNN (Conv2D -> ReLU -> MaxPool).
    * **Result:** The model demonstrated **High Overfitting**.
        * Training Accuracy: ~99%
        * Validation Accuracy: ~88%
    * *Observation:* Without pre-trained knowledge, the model memorized the training set but struggled to generalize.
    
* **Approach 2: Transfer Learning (The Solution)**
    * **Model:** **MobileNetV2** (Pre-trained on ImageNet).
    * **Technique:** Froze the backbone, replaced the classifier head, and fine-tuned for 5 epochs.
    * **Result:** **97% Validation Accuracy**.
    * *Redemption:* The *Apple Cedar Rust* class F1-score jumped from **0.00 (Random Forest)** to **1.00 (MobileNet)**.

---

## 📊 Results Summary

| Model Architecture | Accuracy | Key Observation |
| :--- | :--- | :--- |
| Random Forest (Week 2) | 68.53% | Failed to detect specific disease patterns (0% recall on some classes). |
| Simple CNN (Week 3) | ~88% | High overfitting; unstable validation loss. |
| **MobileNetV2 (Week 3)** | **97.00%** | **Robust, stable, and solved "invisible" classes.** |

### **Visualizations**
*(Placeholder: Upload screenshots of your Loss Curves and Confusion Matrix here)*

![Confusion Matrix](path/to/your/confusion_matrix_image.png)
*Fig 1: Confusion Matrix of MobileNetV2 showing near-perfect diagonal alignment.*

---

## 🚀 How to Run
The project requires **Python 3.x** and the following libraries:
* `torch` / `torchvision` (PyTorch)
* `scikit-learn`
* `matplotlib` & `seaborn`
* `pandas` & `numpy`

1.  Clone the repo:
    ```bash
    git clone [INSERT YOUR REPO LINK HERE]
    ```
2.  Open the notebooks in Jupyter or Kaggle.
3.  Ensure the PlantVillage dataset is mounted in the input directory.

---

## 🏆 Acknowledgments
* **Winter in Data Science (WiDS)** for the mentorship and project structure.
* **Kaggle** for the dataset and GPU resources.
