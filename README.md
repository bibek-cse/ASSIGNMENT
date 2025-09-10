# Fake vs. Real Audio Classification

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![sklearn](https://img.shields.io/badge/scikit--learn-✓-orange)](https://scikit-learn.org/stable/)
[![pandas](https://img.shields.io/badge/pandas-✓-blue)](https://pandas.pydata.org/)
[![shap](https://img.shields.io/badge/SHAP-✓-purple)](https://shap.readthedocs.io/en/latest/)

A comprehensive analysis of classical machine learning models and dimensionality reduction techniques for detecting fake audio clips. This project includes in-depth Exploratory Data Analysis (EDA), model comparison, and Explainable AI (XAI) to interpret model behavior.

---

## Run Online

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bibek-cse/ASSIGNMENT/blob/main/Fake_vs_Real_Audio_Classification.ipynb)  
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/bibekcse/apr-assignment/notebook)

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Concepts & Algorithms](#-key-concepts--algorithms)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Results & Key Findings](#-results--key-findings)
  
---

## Project Overview

This project tackles a binary classification problem: distinguishing between **FAKE** and **REAL** audio clips based on a set of pre-extracted acoustic and spectral features. The core of the analysis is a comparative study of classical machine learning models (Logistic Regression, K-Nearest Neighbors) and the impact of different dimensionality reduction techniques (PCA, SVD, LDA).

A key feature of this project is the integration of **Explainable AI (XAI)** using the SHAP library to interpret model predictions, providing crucial insights into which audio features are most influential in the classification process.

---

## Dataset  

**DEEP-VOICE** --> [DeepFake Voice Recognition Dataset Link](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)

The dataset is a CSV file where each row represents an audio clip, characterized by a collection of statistical features.

-   **Features**: A variety of features describing the audio's spectral and temporal characteristics, such as `mean`, `std`, `spectral_centroid`, `mfcc`, and more.
-   **Target Variable**: The `LABEL` column, where `1` signifies a **FAKE** audio clip and `0` signifies a **REAL** one.
-   **Characteristics**: This is a **balanced binary classification** task, meaning the number of FAKE and REAL samples is approximately equal.

---

## Project Workflow

The analysis is structured into a logical sequence of steps:

1.  **Exploratory Data Analysis (EDA)**: A deep dive into the dataset using statistical summaries, correlation heatmaps, and visualizations of feature distributions and "audio fingerprints" (Radar Plots) to understand data patterns.
2.  **Data Preprocessing**: Standard workflow including train/test splitting and feature scaling using `StandardScaler` to normalize the data.
3.  **Dimensionality Reduction**: Application and comparison of three distinct techniques (PCA, SVD, LDA) to transform the feature space.
4.  **Model Training & Evaluation**: Training `Logistic Regression` and `K-Nearest Neighbors` classifiers on four different data representations (original scaled, PCA, SVD, and LDA). Performance is compared using a suite of metrics (Accuracy, Precision, Recall, F1-Score, AUC).
5.  **Explainable AI (XAI)**: Leveraging the SHAP library to explain the predictions of the best-performing model, providing insights into *why* the model makes certain decisions.
6.  **Conclusion**: Summarizing the findings and concluding with the most effective modeling strategies for this problem.

---

## Key Concepts & Algorithms

This project leverages several fundamental machine learning concepts.

<details>
<summary><b>Dimensionality Reduction Techniques</b> (Click to expand)</summary>

Dimensionality reduction is the process of reducing the number of input features. This is useful for improving model performance, reducing computational cost, and mitigating the "curse of dimensionality."

-   **Principal Component Analysis (PCA)**: An **unsupervised** technique that finds orthogonal (uncorrelated) components that capture the maximum variance in the data. By keeping only the top `k` components, we retain most of the information in a lower-dimensional space.

-   **Singular Value Decomposition (SVD)**: A matrix factorization technique used via `TruncatedSVD`. It is similar to PCA but can be more numerically stable and works well with sparse data.

-   **Linear Discriminant Analysis (LDA)**: A **supervised** technique that finds the feature subspace that maximizes the separability between classes. It aims to maximize the distance between class means while minimizing the variance within each class.

</details>

<details>
<summary><b>Classification Models</b> (Click to expand)</summary>

-   **Logistic Regression**: A linear model that uses a sigmoid function to output a probability for binary classification. It is highly interpretable and serves as a strong baseline.

-   **K-Nearest Neighbors (KNN)**: A non-parametric, instance-based algorithm that classifies a new data point based on the majority class of its `k` nearest neighbors in the feature space.

</details>

---

## Installation

To set up the project environment, follow these steps. It is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bibek-cse/ASSIGNMENT.git
    cd Assignment
    ```

2.  **Install the required libraries:**
    A `requirements.txt` file is included for easy setup.
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file, you can install the packages manually:*
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn shap librosa
    ```

---

## How to Run

1.  Place your dataset CSV file in the designated `data/` directory (or update the path in the notebook).
2.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook "Fake_vs_Real_Audio_Classification.ipynb"
    ```
3.  Execute the cells in the notebook sequentially from top to bottom.

---

## Results & Key Findings

The models were evaluated on four different data representations, and the results provide clear insights into the most effective strategies for this classification task.

- **K-Nearest Neighbors is the Top Performer:** The KNN model, when trained on the original, full-feature scaled data, was the standout performer. It achieved the highest scores across the board (F1-Score: 0.993, AUC: 0.999), indicating near-perfect predictive power.

- **Nuanced Impact of Dimensionality Reduction:** The effect of reducing dimensionality varied significantly between models. For K-Nearest Neighbors, dimensionality reduction with PCA/SVD was highly effective, retaining most of its predictive power (F1-Score ~0.984) with just 10 features. In contrast, for Logistic Regression, PCA/SVD led to a notable drop in performance (F1-Score fell from 0.907 to 0.819). LDA, which reduces the data to a single feature, provided strong results for both models, outperforming PCA/SVD for Logistic Regression.

- **Top Performing Strategy Redefined:** For maximum accuracy, the optimal strategy is K-Nearest Neighbors on the original scaled data. For applications where computational efficiency is critical, K-Nearest Neighbors on PCA-transformed data offers an excellent trade-off, achieving ~99% of the top performance with a fraction of the features.

- **XAI Insights:** The SHAP analysis, when applied to a model, successfully identifies the most influential audio features. This provides valuable domain insights into the acoustic qualities that differentiate real from fake audio, helping to understand the "why" behind the predictions.

---
