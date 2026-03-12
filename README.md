# 🧠 Introduction to Machine Learning (IML) - Project Portfolio

Welcome to this repository containing a series of 5 practical projects completed as part of the Introduction to Machine Learning (IML) course. This repository includes the source code and analysis reports exploring various Machine Learning paradigms, ranging from fundamental theory to generative models and Deep Learning.

## 📋 Table of Contents
- [Projects Overview](#projects-overview)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)

---

## 🚀 Projects Overview

### 📘 Exercise 1: Fundamentals of Machine Learning Theory
This project focuses on the theoretical foundations of machine learning, analyzing the Empirical Risk Minimization (ERM) algorithm.
* **Key Concepts:** Study of approximation error, estimation error, and the generalization gap across different scenarios.
* **Bias-Complexity Tradeoff:** Practical demonstration showing that a restricted hypothesis class has high bias but low estimation error, whereas a highly complex class reduces approximation error but promotes overfitting on small datasets.

### 📗 Exercise 2: k-Nearest Neighbors (kNN) & Boosting
An exploration of instance-based learning and ensemble methods.
* **k-NN & Anomaly Detection:** Evaluation of L1 (Manhattan) and L2 (Euclidean) distance metrics. Results show that a high $k$ smooths decision boundaries, while a low $k$ overfits to local noise. The algorithm was also successfully used to isolate anomalies.
* **Boosting:** Implementation of boosting using decision stumps. Although boosting effectively focuses on hard-to-classify points, the simplicity of the stumps limits perfect accuracy on complex geographical data.

### 📙 Exercise 3: Linear Models and Decision Trees
This module covers classical supervised learning and mathematical optimization.
* **Ridge & Logistic Regression:** Tuning the regularization parameter $\lambda$ to balance model complexity (best result achieved with $\lambda=2$). Both models showed similar performance due to their linear nature.
* **Optimization:** Implementation of Gradient Descent from scratch in NumPy to optimize a 2D function.
* **Decision Trees:** Comparison between shallow trees (max depth = 2) which underfit the data, and deeper trees (max depth = 10) capable of accurately capturing complex non-linear geographical boundaries.

### 📕 Exercise 4: Deep Learning & Transfer Learning
A deep dive into deep neural networks (MLPs and CNNs).
* **Hyperparameters:** Analysis of the impact of Learning Rate, number of epochs (to observe overfitting), and batch size on training stability.
* **Architecture & Batch Normalization:** Study of network depth and width. To address the Vanishing Gradient problem in very deep networks, adding Batch Normalization layers successfully stabilized and accelerated training.
* **Transfer Learning:** Fine-tuning a pre-trained ResNet18 model, compared to training "from scratch" and "Linear Probing" (freezing layers). Results were also benchmarked against the XGBoost algorithm.

### 📓 Exercise 5: Generative Models (Clustering & Text Generation)
The final exercise covers unsupervised learning and generative AI.
* **Mixture Models:** Comparison between Gaussian Mixture Models (GMM) and Uniform Mixture Models (UMM). GMMs, thanks to their elliptical flexibility, adapt perfectly to irregular data contours. Conversely, UMMs struggled due to saturated gradients and shrinking clusters.
* **Initialization Strategies:** Mean initialization based on real coordinates provided smoother convergence compared to random initialization.
* **Text Generation:** Evaluation of a Transformer model's capabilities over 10 epochs. Using the *Top-k* sampling method generated much more coherent and diverse sentences than the standard approach.

---

## 💻 Technologies Used
* **Language:** Python
* **Machine Learning & Data Science:** Scikit-learn, NumPy, Pandas, XGBoost
* **Deep Learning:** PyTorch (ResNet18, MLPs, Transformers)
* **Data Visualization:** Matplotlib, Seaborn

