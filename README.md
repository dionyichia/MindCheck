# MindScope: Mental Health Predictor

**Team:** REP2 Group 1
- Chia Dion Yi, U2321724K
- Isaac Wong Vun Hau, U2322857K
- Wang RuiYun, U2322576B

**Course:** SC1015 Mini Project

[Link to Presentation Video](https://youtu.be/GDiBVJ74Snk)

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Problem Definition](#1-problem-definition-10)
3.  [Data Preparation and Cleaning](#2-data-preparation-and-cleaning-10)
4.  [Exploratory Data Analysis (EDA) / Visualization](#3-exploratory-data-analysis-eda--visualization-20)
5.  [Use of Machine Learning Techniques](#4-use-of-machine-learning-techniques-20)
6.  [Data-Driven Insights and Recommendations](#5-data-driven-insights-and-recommendations-20)
7.  [Quality of Final Team Presentation](#6-quality-of-final-team-presentation-and-overall-impressions-10)
8.  [Learning Something New / Beyond Course](#7-learning-something-new-and-doing-something-beyond-this-course-10)

---

## Project Overview

Mental health is a critical concern, particularly among university students, with studies indicating that a significant percentage of individuals live with mental disorders, and approximately 50% of cases go undiagnosed. Early detection and intervention can significantly improve outcomes. This project, MindScope, aims to explore the potential of machine learning techniques to predict the likelihood of depression and anxiety in university students, potentially serving as an early-stage indicator. We leverage a multi-modal approach, analyzing structured survey data (Parts A & B) and unstructured text data (Part C) to build predictive models.

---

### 1. Problem Definition

**Motivation:** The high prevalence of undiagnosed mental health conditions like depression and anxiety among university students poses a significant challenge. Timely identification is crucial but often difficult.

**Problem Statement:** Can we develop a machine learning system that utilizes diverse data sources – specifically, structured survey responses (capturing demographic, academic, lifestyle factors) and unstructured free-text descriptions of daily experiences – to accurately predict the likelihood of an individual student experiencing depression or anxiety?

**Goal:** To build and evaluate machine learning models capable of processing these different data types and fusing their predictions to provide separate likelihood scores (0-100%) for depression and anxiety, potentially aiding in early detection efforts.

**Datasets Used:**
1.  **(Part A) Depression Dataset:** Contains structured data (numeric, categorical, binary features like academic pressure, sleep duration, etc.) from 27,901 entries, with a binary target variable indicating depression.
2.  **(Part B) Anxiety Dataset:** Contains structured data (numeric, categorical features like isolation, future insecurity, etc.) from 86 entries, with a binary target variable indicating high/low anxiety (derived from original ratings).
3.  **(Part C) Sentiment Analysis Dataset:** Contains unstructured text data (student write-ups) along with categorical labels from 53,043 entries, originally with multiple mental health statuses. The target for our model was refined to predict 'Normal', 'Anxiety', or 'Depression'.

---

### 2. Data Preparation and Cleaning

* **(Part A) Depression Dataset:**
    * Handled missing values by replacing nulls using the mean for numeric columns and the mode for categorical columns.
    * Encoded categorical columns using Label Encoding.
    * Normalized numeric and encoded categorical variables using `MinMaxScaler` to scale features between 0 and 1, suitable for the ANN model.
* **(Part B) Anxiety Dataset:**
    * Checked for missing values.
    * Performed data type conversions (e.g., converting columns to strings) and split comma-separated values in columns like `stress_relief_activities`.
    * Converted object-type range columns (e.g., `cgpa`, `average_sleep`) to numeric averages to facilitate analysis and visualization.
    * Encoded categorical features.
    * Reclassified the target variable 'Anxiety' into High/Low based on a threshold (identified as 4 for better balance).
    * Applied feature scaling using `StandardScaler` to standardize the input data (mean=0, std=1). This is crucial for distance-based algorithms like SVM, ensuring all features contribute equally.
    * Addressed class imbalance using upsampling: the minority class was resampled *with replacement* until it matched the size of the majority class.
* **(Part C) Sentiment Analysis Dataset:**
    * Initial text cleaning involved steps like lowercasing and potentially removing punctuation.
    * **Class Filtering:** Based on EDA insights and model constraints:
        * Removed 'Suicidal' status due to LLM safeguard limitations.
        * Removed 'Personality Disorder' and 'Bipolar' statuses due to high linguistic similarity (cosine similarity > 0.85-0.91 with Depression/Normal based on TF-IDF) and small sample sizes, potentially confusing the classifier.
    * **Final Classes:** Focused the classification task on the remaining statuses: 'Normal', 'Anxiety', and 'Depression'.
    * **Dataset Balancing:** Addressed class imbalance in the filtered dataset by downsampling the majority classes to match the size of the smallest class (`min_class_size`), ensuring equal representation during fine-tuning (`balanced_df = df.groupby("status", group_keys=False).apply(lambda x: x.head(min_class_size))`). The balanced dataset was saved (`balanced_dataset.csv`).
    * **Formatting for LLM:** Prepared the data in the required format for the `SFTTrainer`, involving specific input/output text structures (e.g., "statement: [student's text] label: [status]").

---

### 3. Exploratory Data Analysis (EDA) / Visualization 

* **(Part A) Depression Dataset:**
    * Analyzed univariate distributions of features.
    * Calculated correlations between features and the 'Depression' target using **point-biserial correlation** (for numeric/categorical vs. binary) and **phi-squared coefficient** (for binary vs. binary). Found no single strong predictor, suggesting a complex multi-factor relationship.
    * Performed linearity analysis using regression plots (with `logistic=True`) to visualize relationships and assess suitability for models like logistic regression, noting various relationship shapes (flat, linear, sigmoidal).
    * Conducted outlier analysis using boxplots.
* **(Part B) Anxiety Dataset:**
    * Analyzed feature correlations with the 'Anxiety' target using point-biserial correlation and phi-squared coefficient.
    * Identified 'Isolation', 'Future Insecurity', and 'Social Relationship' (negative) as top correlates (excluding 'Depression' itself to avoid circular logic).
    * Visualized data distributions to determine the threshold for reclassifying anxiety levels and assess class balance.
* **(Part C) Sentiment Analysis Dataset:**
    * **Initial Analysis:** Visualized word frequencies (using word clouds and frequency plots) for each original status, revealing many common, non-discriminatory words (e.g., 'i', 'to', 'and', 'my', 'like', 'want').
    * **TF-IDF Analysis:** Applied TF-IDF with unigrams and n-grams (bigrams, trigrams) to identify potentially more important terms per status. While some distinct terms emerged, **significant overlap persisted**, particularly between 'Personality Disorder' & 'Depression' and 'Bipolar' & 'Depression'.
    * **Similarity Analysis:** Quantified inter-class similarity using cosine similarity on TF-IDF vectors and further validated using log-likelihood analysis on distinctive words. This confirmed the high linguistic similarity between certain statuses (e.g., Personality Disorder vs. Normal/Bipolar/Depression), supporting the decision to filter these classes.
    * **Conclusion:** EDA highlighted the limitations of relying solely on surface-level lexical features (like TF-IDF) for this nuanced task, motivating the shift to a deep learning approach (LLM fine-tuning) capable of capturing semantic meaning and context.

---

### 4. Use of Machine Learning Techniques

* **(Part A) Depression Prediction (ANN):**
    * **Model:** An Artificial Neural Network (ANN) was selected for the large (27,901 samples) Depression dataset.
    * **Architecture:** Input layer (13 nodes corresponding to features), 2 hidden layers (16 and 8 nodes respectively) using **ReLU activation function**, and an output layer (1 node) using **Sigmoid activation function** for binary probability output.
    * **Training:** Loss calculated using **Binary Cross-Entropy**, with weights updated using the **Adam optimizer**.
    * **Rationale:** ANN's ability to capture complex, non-linear interactions between multiple features, suitable given the EDA finding that individual features had low correlation but might be predictive in combination.
    * **Performance:** Achieved an **Accuracy of 0.84** and **ROC AUC of 0.91**.
* **(Part B) Anxiety Prediction (SVM):**
    * **Model Comparison:** Evaluated four models (Logistic Regression, Random Forest, XGBoost, SVM) for the small (86 samples) Anxiety dataset.
    * **Model Choice:** Selected **Support Vector Machine (SVM) with an RBF kernel**.
    * **Rationale:** SVMs can perform well on smaller datasets and find non-linear boundaries (with RBF kernel). It outperformed other tested models on the validation set based on performance and resistance to overfitting.
    * **Preprocessing:** Applied `StandardScaler` (crucial for distance-based SVM) and upsampling (minority class with replacement).
    * **Fine-tuning:** Implemented **Probability Threshold Tuning** by optimizing the decision threshold on the precision-recall curve to maximize the F1-score, resulting in an optimal threshold of **0.382** (instead of the default 0.5).
    * **Performance:** Achieved an **Accuracy of 0.85** and **ROC AUC of 0.792**.
* **(Part C) Text-Based Status Prediction (Fine-tuned LLM):**
    * **Model Choice:** Fine-tuned `meta-llama/Meta-Llama-3.1-8B-Instruct`, a powerful pre-trained Large Language Model. Selected for its balance of performance and computational feasibility.
    * **Efficiency Techniques:** Employed 4-bit quantization using `bitsandbytes` integrated via the `unsloth` library for significantly reduced memory footprint and faster training. Utilized Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).
    * **Training:** Used the `SFTTrainer` from the `trl` library. Trained for 1 epoch on the prepared, balanced dataset containing 'Normal', 'Anxiety', and 'Depression' classes.
    * **Rationale:** Chosen to leverage the LLM's pre-trained understanding of language nuances, context, and semantics, overcoming the limitations of keyword-based methods identified in EDA.
    * **Performance:** Achieved a significant improvement in predictive accuracy, increasing from a baseline accuracy of approximately 73% to an average accuracy of **~95.5%** post-fine-tuning on the test set.

    [Link to Fine-Tuned Model](https://huggingface.co/fiendfrye/mental-status-classifier-lama-3.1-8b-fine-tuned)

---

### 5. Data-Driven Insights and Recommendations

* **(Part A & B) Insights:** EDA on structured data revealed the multi-factorial nature of depression and anxiety, with no single dominant predictor. Correlation analysis identified key factors associated with anxiety (Isolation, Future Insecurity, Social Relationship).
* **(Part C) Insights:** Text analysis (EDA) highlighted significant linguistic overlap between different mental health statuses when using traditional methods like TF-IDF. This underscored the need for models that understand semantics and context. Fine-tuning the Llama 3.1 model proved highly effective, demonstrating that LLMs can capture subtle linguistic cues indicative of different mental states ('Normal', 'Anxiety', 'Depression') much better than lexical methods.
* **Model Performance Summary:**
    1.  (Part A) The ANN achieved **Accuracy: 0.84, ROC AUC: 0.91** on the depression task.
    2.  (Part B) The SVM achieved **Accuracy: 0.85, ROC AUC: 0.792** on the anxiety task after threshold tuning.
    3.  (Part C) The fine-tuned LLM achieved **~95.5%** average accuracy on the text classification task.
* **Output Fusion/Recommendations:** The final "MindScope" system integrates the predictions from the three separate models.
    1.  (Part A) ANN outputs a likelihood score (0-1) for depression.
    2.  (Part B) SVM outputs a likelihood score (0-1) for anxiety.
    3.  (Part C) Fine-tuned LLM predicts a class ('Normal', 'Anxiety', 'Depression') from text.
    4.  A **Fusion Layer** combines these: The LLM's categorical prediction provides contextual grounding, acting as a 'nudge' (multiplier: 1.15 for Anxiety/Depression, 0.90 for Normal) applied to the respective likelihood scores from the ANN/SVM. The final outputs are scaled (MinMax) to remain within the 0-1 range, representing the final predicted likelihoods for depression and anxiety based on combined evidence.

---

### 6. Final Team Presentation and Overall Impressions

* Our presentation aimed for clarity, logical flow, and effective visualization of our process and results. We sought to demonstrate a clear understanding of the problem, methods, and outcomes.
* [Link to Presentation Slides](./presentation_slides.pdf)

---

### 7. Learning Something New and Doing Something Beyond This Course

* **(Part A & B Focused):**
    * **Correlation Techniques:** Learned and applied specific correlation methods suitable for mixed data types: **Point-Biserial correlation** (numeric/categorical vs. binary) and **Phi-Squared coefficient** (binary vs. binary).
    * **ANN Implementation:** Gained practical experience in designing an ANN, including selecting appropriate **activation functions (ReLU, Sigmoid)** and **loss functions (Binary Cross-Entropy)** based on the task.
    * **Data Handling:** Practiced techniques for handling imbalanced datasets (**upsampling with replacement**) and the importance and application of different **feature scaling** methods (`MinMaxScaler`, `StandardScaler`) with rationale.
    * **Model Evaluation & Tuning:** Explored different classification models (Logistic Regression, RF, XGBoost, SVM) and learned techniques like **Probability Threshold Tuning** using precision-recall curves to optimize for F1-score.
* **(Part C Focused - Advanced NLP & LLMs):**
    * **LLM Fine-Tuning:** Went significantly beyond basic NLP by learning and implementing the full fine-tuning pipeline for a state-of-the-art Large Language Model (Llama 3.1 8B).
    * **Efficiency Techniques:** Gained hands-on experience with crucial techniques for managing LLMs on limited hardware: **4-bit quantization** (`bitsandbytes`, `unsloth`) and **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**.
    * **Modern NLP Libraries:** Utilized advanced libraries like `transformers`, `peft`, and `trl` (specifically `SFTTrainer`) for model loading, configuration, and training.
    * **LLM Architecture Awareness:** Explored the architectural advancements in modern decoder-only LLMs (like Llama 3.1's use of RMSNorm, RoPE, GQA, SwiGLU) compared to the original Transformer.
* **(Overall):**
    * **Model Fusion:** Designed and implemented a mechanism to combine outputs from distinct models operating on different data modalities (structured and unstructured).

---
