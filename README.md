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
Evaluate the limits of non-deep-learning approaches & find the "ceiling" of classical Machine Learning. Build a system that classifies plant diseases using "Old School" methods (SVMs, Random Forests) on raw pixel data. This score will serve as the benchmark that our Deep Learning models must beat in the week that follows.

#### Feature Engineering
- Images were resized from `(224, 224, 3)` to `(64, 64, 3)` and flattened into a feature matrix with corresponding class labels. This was done to ease the load on RAM & also faster & efficient training. 
- Data was split into training and testing sets and standardized using z-score normalization via StandardScaler.
- Dimensionality reduction was performed using fast randomized **Principal Component Analysis (PCA)**, reducing the feature space from `12,288` to `150` components.

#### Models Evaluated
- **DummyClassifier** (Chance baseline): 10.58% accuracy
- **Nearest Centroid**: 37.28% accuracy
- **Linear SVM (PCA+Linear)**: 66.18% accuracy
- **Random Forest Classifier**: 68.53% accuracy (best classical model)

#### Failure Analysis
- Random Forest performs well when diseases have distinct visual signatures or consistent texture/color cues. On raw pixels, it handles coarse texture and strong color cues well.
- Several classes exhibit near-zero recall due to strong visual similarity with related diseases and class imbalance, highlighting the inability of pixel-based Random Forests to separate subtle disease variations.
- Healthy leaves are frequently misclassified because “healthy” varies significantly across plant species, making a single healthy label biologically inconsistent in raw pixel feature space.
- Diseases within the same plant species are heavily confused as they differ primarily in fine-grained lesion morphology rather than global color, which flattened pixel representations fail to capture.

**Conclusion**  
Classical machine learning models on flattened pixel features achieve moderate performance (~69% accuracy) and capture coarse color and texture cues but fail on fine-grained, visually similar diseases and several minority classes. The confusion patterns, especially within the same plant species and between healthy vs. diseased leaves highlight fundamental limitations of non-spatial models. These results establish a solid baseline and clearly motivate the need for CNN-based approaches to learn localized, disease-specific visual features for meaningful performance gains. 

---

### Week 3: Deep Learning & Transfer Learning

**Goal**  
Build a CNN that learns features directly from images and outperform classical ML, then apply transfer learning to achieve ~85–90% accuracy, marking the transition from traditional methods to modern, powerful deep learning.

---

#### Stage 1: Simple CNN (From Scratch)
The dataset was split into augmented training and clean validation subsets using an 80–20 split. DataLoaders were created for batched training and validation, and augmented samples were visually inspected.
**Architecture**
- A deliberately *minimal* 2-layer CNN:
  - `Conv2D (3 → 16 filters, 3×3) → ReLU → MaxPool (2×2)`
  - `Conv2D (16 → 32 filters, 3×3) → ReLU → MaxPool (2×2)`
  - `Flatten → Dense (32 × 56 × 56 → 38)`

**The Overfitting Experiment**
- The model was trained using mini-batch gradient descent with backpropagation for 10 epochs and evaluated on a validation set each epoch.
- Using Adam and cross-entropy loss, the CNN learned effective visual features.
- Results:
  - Training Accuracy: **~99.5%**
  - Validation Accuracy: **~88%**

**Key Observation**
- Loss curves showed classic divergence:
  - Training loss ↓
  - Validation loss ↑  
  → Clear evidence of overfitting

---

#### Stage 2: Transfer Learning (The Solution)

**Architecture**
- A **MobileNetV2** model pretrained on ImageNet was initialized, its backbone frozen, and the classifier head replaced to adapt the network for plant disease classification.
**Technique**
- Frozen Backbone reuses pretrained feature extractors.
- Custom Classification Head is a linear layer for 38 classes.
- Data Augmentation is still present in the form of random rotations, horizontal/vertical flips, affine zoom transformations.

**Result**
- **97% validation accuracy** in just **5 epochs**.
- Stable convergence with no train–validation gap.

---

## 📊 Results Summary

| Model Architecture | Accuracy | Key Observation |
| :--- | :--- | :--- |
| Random Forest (Week 2) | 68.53% | Failed to detect specific disease patterns (0% recall on some classes) |
| Simple CNN (Week 3) | ~88% | High overfitting; unstable validation loss |
| Transfer learning with **MobileNetV2 (Week 3)** | **97.34** | **Robust, stable, and solved "invisible" classes** |

---

## 📈 Performance Visualizations

### 1. Overfitting in Simple CNN (The Problem)
Training accuracy quickly approaches 100% while validation accuracy stagnates around ~87–88%, alongside decreasing training loss and increasing validation loss.This indicates that the Simple CNN is memorizing training samples rather than learning robust, transferable visual features, leading to limited generalization on unseen data. And this makes it a classic case of overfitting and poor generalization.
### 2. Robustness in Transfer Learning (The Solution)
MobileNetV2 exhibits stable convergence with training and validation curves converge rapidly with steadily decreasing loss and consistently higher validation accuracy, implying good generalization without overfitting. The confusion matrix shows near-perfect diagonal dominance, indicating minimal misclassification and strong class-wise separability across all 38 diseases

---

## ✅ Key Takeaways

- EDA-driven insights strongly influence downstream modeling decisions
- Classical ML models are insufficient for high-dimensional image data
- CNNs learn spatial hierarchies but require regularization
- Transfer Learning provides both **performance and efficiency**, even with limited training epochs
