# Week 1 Learnings — PlantVillage Dataset (EDA)

This document captures key observations, insights, and unexpected findings from the exploratory data analysis (EDA) of the PlantVillage dataset. The focus is on aspects that influence problem difficulty, modeling decisions, and potential risks during training and deployment.


## 1. Class Distribution and Representation

A significant variation exists in the number of images across disease classes. While some classes are heavily represented, several diseases have noticeably fewer samples. This imbalance was more pronounced than initially expected and highlights the risk of biased model learning toward majority classes if no corrective measures (e.g., class weighting or augmentation) are applied.

**Surprise:**  
Despite being a widely used benchmark dataset, PlantVillage is not uniformly balanced across disease categories.


## 2. Visual Similarity Between Disease Classes

Diseases belonging to the same plant species often exhibit strong visual similarity, especially in terms of leaf texture, discoloration patterns, and lesion shapes. In multiple cases, distinguishing between diseases was challenging even through manual inspection of multiple samples.

**Key Insight:**  
The classification task is inherently fine-grained, requiring models to learn subtle visual cues rather than relying on coarse features.


## 3. Image Resolution and Structural Consistency

Most images share a consistent resolution and framing, suggesting standardized data collection practices. This consistency simplifies preprocessing pipelines but also reduces natural variability in the data.

**Surprise:**  
The lack of significant resolution diversity may encourage overfitting to dataset-specific characteristics rather than learning robust disease features.


## 4. Background Uniformity and Data Bias

A majority of images have clean, homogeneous backgrounds with centrally positioned leaves. While this improves visual clarity, it introduces a strong dataset bias.

**Key Risk Identified:**  
Models trained on this dataset may rely on background cues or framing regularities (shortcut learning), potentially limiting generalization to real-world field images with complex backgrounds.


## 5. Image Quality and Lighting Conditions

Overall image quality is high, with minimal blur and relatively uniform lighting conditions across samples. Extreme variations in brightness or contrast are rare.

**Observation:**  
While beneficial for controlled experimentation, this uniform quality reduces exposure to real-world noise, which could affect deployment robustness.


## 6. Overall Takeaway

The PlantVillage dataset is clean, structured, and suitable for benchmarking plant disease classification models. However, the combination of class imbalance, strong visual similarity between diseases, and controlled imaging conditions makes the task both deceptively challenging and potentially prone to overfitting. These insights will directly inform preprocessing strategies, augmentation design, and evaluation protocols in later stages of the project.

---
