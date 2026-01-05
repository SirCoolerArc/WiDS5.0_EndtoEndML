# 🌱 Plant Disease Detection | End-to-End ML Systems | WiDS 5.0

**Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)

---

## 📌 Project Overview
This project explores the progression from traditional Machine Learning to state-of-the-art Deep Learning techniques for classifying plant leaf diseases. Working with the **PlantVillage dataset**, the mid-term goal was to build a robust model capable of assisting in disease diagnosis for different crops like Apple, Corn, Potato, Tomato, etc.

All of the code is stored in three different Kaggle notebooks, each of which can be accessed as they are linked to this repository. 

The project follows a **First Principles approach** along a 3-week timeline :

- **Week 1**: Understanding the data through rigorous Exploratory Data Analysis (EDA).
- **Week 2**: Establishing baselines using Classical Machine Learning techniques like SVMs & Random Forests.
- **Week 3**: Developing state-of-the-art Deep Learning models (CNNs & Transfer Learning).

---

## 📂 Repository Structure

| Notebook | Description | 
|--------|------------|
| `Week1_EDA.ipynb` | Exploratory Data Analysis: data inspection, class imbalance visualization, pixel intensity analysis | 
| `Week2_ClassicalML.ipynb` | Classical ML baselines: feature extraction, PCA, SVM vs Random Forest | 
| `Week3_CNNs.ipynb` | Deep Learning: custom CNN (from scratch) vs Transfer Learning (MobileNetV2) | |

---

## 🗓️ Weekly Methodology & Technical Details

### Week 1: Exploratory Data Analysis (EDA)

**Goal**  
Peform EDA to get at least class distribution visualization, sample image grid (multiple images per class), Image resolution/size analysis and 2-3 meaningful insights about the dataset.

**Key Decisions & Analysis**

- **Data Selection**:
Chose the *color* folder in the dataset to preserve diagnostic features such as chlorosis and rust spores that are lost in grayscale images.
- **Class Distribution Analysis**:
Various different graphs & visualisations were plotted like bar graphs, pie chart, sun burst chart, stacked bar graphs, treemap to assess the distribution pattern in the dataset.

- **Data Quality Audit**:
   - Aspect ratio was checekd using PIL, where all the images were found to be of the standard 256×256 size.
   - Background consistency check was done using corner-pixel color sampling.
   - Blurs were detected using the concept of Laplacian blur for various classes & plotted into a candlestick chart.
   - Lighting variation analysis was done using HSV-based brightness (V-channel) distribution across classes.
   - Leaf-to-background ratio check was done using grayscale conversion and Otsu’s thresholding to estimate leaf pixel coverage.
   
- **Visual inspection**:
  This was performed using class-wise galleries and side-by-side comparisons.
  
- **Biological Feature Analysis**:
   - Chlorophyll variation was analyzed using green-channel intensity distributions across biologically diverse classes.
   - Biological color separability was evaluated using mean HSV hue and saturation features across healthy, viral, and fungal classes.
   - Leaf texture differences were analyzed using GLCM-based contrast and homogeneity features.
   - Full RGB color distribution was analyzed using mean channel intensities across disease classes.
   - Model feasibility was evaluated using PCA on flattened pixel intensities to assess class separability.
   - Non-linear class separability was analyzed using t-SNE embeddings of scaled pixel features.

**Key Insight**  
- Massive class imbalance observed:
  - `Orange___Huanglongbing`: ~5.5K images  
  - `Corn_(maize)___healthy`: 21 images
- Background: Corner pixel analysis confirms a somehwat uniform background.
- Lighting: Brightness distribution is more or less consistent.
- Several disease classes are visually similar, increasing the difficulty of fine-grained classification.

---

### Week 2: Classical Machine Learning Baselines

**Goal**  
Evaluate the limits of non-deep-learning approaches by treating images as numerical feature vectors.

#### Feature Engineering
- Flattened images from `(224, 224, 3)` to vectors of size **150,528**
- Applied **StandardScaler** to normalize pixel intensities
- Experimented with **PCA (Principal Component Analysis)** for dimensionality reduction

#### Models Evaluated
- **DummyClassifier** (Chance baseline): ~10% accuracy
- **Linear SVC**: ~66% accuracy
- **Random Forest Classifier**: **68.5% accuracy** (best classical model)

#### Failure Analysis
- Random Forest failed catastrophically on texture-heavy classes:
  - *Apple Cedar Rust*: **F1-score = 0.00**

**Conclusion**  
Classical ML models struggle to capture **spatial dependencies** (edges, shapes, textures) inherent in image data.

---

### Week 3: Deep Learning & Transfer Learning

**Goal**  
Move from manual feature engineering to **automatic feature learning**.

---

#### Stage 1: Custom CNN (From Scratch)

**Architecture**
- A deliberately *minimal* 2-layer CNN:
  - `Conv2D (16 filters) → ReLU → MaxPool`
  - `Conv2D (32 filters) → ReLU → MaxPool`
  - `Dense (Output Layer)`

**The Overfitting Experiment**
- Trained **without data augmentation**
- Results:
  - Training Accuracy: **~99%**
  - Validation Accuracy: **~88%**

**Key Observation**
- Loss curves showed classic divergence:
  - Training loss ↓
  - Validation loss ↑  
  → Clear evidence of overfitting

---

#### Stage 2: Transfer Learning (The Solution)

**Architecture**
- **MobileNetV2**, pretrained on ImageNet

**Technique**
- **Frozen Backbone**: reused pretrained feature extractors
- **Custom Classification Head**: linear layer for 38 classes
- **Data Augmentation**:
  - Random rotations
  - Horizontal/vertical flips
  - Affine zoom transformations

**Result**
- **97% validation accuracy in just 5 epochs**
- Stable convergence with no train–validation gap

---

## 📊 Results Comparison

| Model Architecture | Accuracy | Apple Cedar Rust (F1) | Potato Healthy (F1) |
|------------------|----------|----------------------|---------------------|
| Random Forest (Week 2) | 68.53% | 0.00 | 0.00 |
| Simple CNN (Week 3) | 88.42% | 0.72 | 0.45 |
| MobileNetV2 (Week 3) | **97.00%** | **1.00** | **0.80** |

---

## 📈 Performance Visualizations

### 1. Overfitting in Simple CNN (The Problem)
Validation loss increases while training loss decreases, indicating memorization rather than generalization.

### 2. Robustness in Transfer Learning (The Solution)
MobileNetV2 exhibits stable convergence with overlapping training and validation curves.

---

## ✅ Key Takeaways

- EDA-driven insights strongly influence downstream modeling decisions
- Classical ML models are insufficient for high-dimensional image data
- CNNs learn spatial hierarchies but require regularization
- Transfer Learning provides both **performance and efficiency**, even with limited training epochs


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

## 🏆 Acknowledgments
* **Winter in Data Science (WiDS)** for the mentorship and project structure.
* **Kaggle** for the dataset and GPU resources.
