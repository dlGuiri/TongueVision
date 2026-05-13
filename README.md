This README provides a comprehensive overview of **TongueVision**, a hybrid deep learning system developed for non-invasive diabetes risk assessment.

# TongueVision: Hybrid Deep Learning for Tongue-Based Diabetes Risk Assessment

TongueVision is an AI-driven screening tool designed to address the growing burden of undiagnosed diabetes, particularly in underserved communities. By analyzing observable features of the tongue—such as color, texture, and coating—the system provides a low-cost, non-invasive alternative to traditional invasive screening methods.

---

## 🚀 Key Features

* **Non-Invasive Screening:** Utilizes tongue imagery as a digital indicator of underlying diabetic conditions.
* **Hybrid AI Architecture:** Combines **ConvNeXt Tiny** for local feature extraction (texture, fissures) with **Swin Transformers** for global contextual reasoning.
* **Adaptive Feature Fusion:** Implements a learnable **Attention-Guided Feature Fusion (AGFF)** mechanism that dynamically weights features using spatial and channel attention.

---

## 🛠️ Tech Stack

* **Backend:** Python API Server
* **AI Engine:** PyTorch framework utilizing pre-trained ImageNet weights and mixed-precision training (AMP)
---

## 🧠 System Architecture

The system follows a dual-branch architecture to process tongue images simultaneously:

1. **Branch 1 (CNN):** Employs hierarchical convolutional operations to capture fine-grained textural patterns and color variations.
2. **Branch 2 (Transformer):** Models long-range dependencies and global relationships across the entire tongue surface.
3. **AGFF Block:** Synergistically fuses these features to highlight diagnostically salient regions while minimizing background noise.

---

## 📊 Dataset & Training

The model was developed using a two-stage training process:

* **Pre-training:** Over 5,000 images from the **Dryad** dataset were used to learn fundamental tongue geometry.
* **Task-Specific Training:** Merged data from **Mendeley** and **DMT (Diabetes Mellitus Tongue)** datasets, consisting of clinical images labeled for diabetes-related manifestations.
* **Augmentation:** Applied geometric (rotations, flips) and photometric (brightness, contrast) scaling to improve generalization and mitigate overfitting.

---

## 📈 Performance Evaluation

The system is benchmarked against homogeneous variants (Dual-ConvNeXt and Dual-Swin) using the following metrics:

* **Accuracy:** Overall prediction correctness.
* **Precision:** Correctness of positive predictions.
* **Recall (Sensitivity):** Ability to identify actual positive cases, which is critical for early diabetes detection.
* **F1-Score:** The harmonic mean of precision and recall for a balanced assessment.

---

## 🛡️ Ethical Considerations

TongueVision is developed with strict adherence to ethical principles:

* **Privacy:** All data is treated with confidentiality and used solely for academic research.
* **Voluntary Participation:** Users are informed of the purpose of the tool and maintain the right to withdraw at any time.
