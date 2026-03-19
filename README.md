[cite_start]This README provides a comprehensive overview of **TongueVision**, a hybrid deep learning system developed for non-invasive diabetes risk assessment[cite: 227, 399].

# TongueVision: Hybrid Deep Learning for Tongue-Based Diabetes Risk Assessment

[cite_start]TongueVision is an AI-driven screening tool designed to address the growing burden of undiagnosed diabetes, particularly in underserved communities[cite: 212, 235]. [cite_start]By analyzing observable features of the tongue—such as color, texture, and coating—the system provides a low-cost, non-invasive alternative to traditional invasive screening methods[cite: 209, 216, 236].


---

## 🚀 Key Features

* [cite_start]**Non-Invasive Screening:** Utilizes tongue imagery as a digital indicator of underlying diabetic conditions[cite: 211, 235].
* [cite_start]**Hybrid AI Architecture:** Combines **ConvNeXt Tiny** for local feature extraction (texture, fissures) with **Swin Transformers** for global contextual reasoning[cite: 402, 406].
* [cite_start]**Adaptive Feature Fusion:** Implements a learnable **Attention-Guided Feature Fusion (AGFF)** mechanism that dynamically weights features using spatial and channel attention[cite: 407, 441].
* [cite_start]**End-to-End Web Platform:** A functional prototype built for automated diabetes risk assessments from user-uploaded images[cite: 229, 410].

---

## 🛠️ Tech Stack

* [cite_start]**Frontend:** React, Next.js[cite: 363, 449].
* [cite_start]**Backend:** Python API Server[cite: 363, 450].
* [cite_start]**AI Engine:** PyTorch framework utilizing pre-trained ImageNet weights and mixed-precision training (AMP)[cite: 363, 444, 447].
* [cite_start]**Database:** MongoDB[cite: 363].

---

## 🧠 System Architecture

[cite_start]The system follows a dual-branch architecture to process tongue images simultaneously[cite: 423]:

1.  [cite_start]**Branch 1 (CNN):** Employs hierarchical convolutional operations to capture fine-grained textural patterns and color variations[cite: 438].
2.  [cite_start]**Branch 2 (Transformer):** Models long-range dependencies and global relationships across the entire tongue surface[cite: 439].
3.  [cite_start]**AGFF Block:** Synergistically fuses these features to highlight diagnostically salient regions while minimizing background noise[cite: 292, 408].


---

## 📊 Dataset & Training

[cite_start]The model was developed using a two-stage training process[cite: 326]:

* [cite_start]**Pre-training:** Over 5,000 images from the **Dryad** dataset were used to learn fundamental tongue geometry[cite: 329, 330, 421].
* [cite_start]**Task-Specific Training:** Merged data from **Mendeley** and **DMT (Diabetes Mellitus Tongue)** datasets, consisting of clinical images labeled for diabetes-related manifestations[cite: 331, 332, 422].
* [cite_start]**Augmentation:** Applied geometric (rotations, flips) and photometric (brightness, contrast) scaling to improve generalization and mitigate overfitting[cite: 339, 342].

---

## 📈 Performance Evaluation

[cite_start]The system is benchmarked against homogeneous variants (Dual-ConvNeXt and Dual-Swin) using the following metrics[cite: 231, 381]:
* [cite_start]**Accuracy:** Overall prediction correctness[cite: 383].
* [cite_start]**Precision:** Correctness of positive predictions[cite: 386].
* [cite_start]**Recall (Sensitivity):** Ability to identify actual positive cases, which is critical for early diabetes detection[cite: 372, 388].
* [cite_start]**F1-Score:** The harmonic mean of precision and recall for a balanced assessment[cite: 389, 390].

---

## 🛡️ Ethical Considerations

TongueVision is developed with strict adherence to ethical principles:
* [cite_start]**Privacy:** All data is treated with confidentiality and used solely for academic research[cite: 459].
* [cite_start]**Voluntary Participation:** Users are informed of the purpose of the tool and maintain the right to withdraw at any time[cite: 458, 460].

---

**Would you like me to help you draft the "Installation" or "Usage" sections for this README based on your current local setup?**
