# MindScope: Mental Health Predictor

**Team:** REP2 Group 1 (Dion, Isaac, Ruiyun)
**Course:** SC1015 Mini Project

## Project Overview

Mental health is a critical concern, particularly among university students, with studies indicating that a significant percentage of individuals live with mental disorders, and approximately 50% of cases go undiagnosed. Early detection and intervention can significantly improve outcomes. This project, MindScope, aims to explore the potential of machine learning techniques to predict the likelihood of depression and anxiety in university students, potentially serving as an early-stage indicator. We leverage a multi-modal approach, analyzing structured survey data and unstructured text data to build predictive models.

---


### 1. Problem Definition (10%)

**Motivation:** The high prevalence of undiagnosed mental health conditions like depression and anxiety among university students poses a significant challenge. Timely identification is crucial but often difficult.

**Problem Statement:** Can we develop a machine learning system that utilizes diverse data sources – specifically, structured survey responses (capturing demographic, academic, lifestyle factors) and unstructured free-text descriptions of daily experiences – to accurately predict the likelihood of an individual student experiencing depression or anxiety?

**Goal:** To build and evaluate machine learning models capable of processing these different data types and fusing their predictions to provide separate likelihood scores (0-100%) for depression and anxiety, potentially aiding in early detection efforts.

**Datasets Used:**
1.  **Depression Dataset:** Contains structured data (numeric, categorical, binary features like academic pressure, sleep duration, etc.) from 27,901 entries, with a binary target variable indicating depression.
2.  **Anxiety Dataset:** Contains structured data (numeric, categorical features like isolation, future insecurity, etc.) from 86 entries, with a binary target variable indicating high/low anxiety (derived from original ratings).
3.  **Sentiment Analysis Dataset:** Contains unstructured text data (student write-ups) along with categorical labels from 53,043 entries, originally with multiple mental health statuses. The target for our model was refined to predict 'Normal', 'Anxiety', or 'Depression'.

### 2. Data Preparation and Cleaning (10%)

*(Briefly describe the steps taken for each dataset)*

* **Depression Dataset:** Handling of missing values (if any), encoding categorical features (e.g., Gender, Dietary Habits) and binary features into numerical formats suitable for the ANN model. Feature scaling was likely applied.
* **Anxiety Dataset:** Encoding categorical features, reclassifying the target variable 'Anxiety' into High/Low based on a threshold (identified as 4 for better balance), applying feature scaling (e.g., StandardScaler), and addressing class imbalance using upsampling (specifically on the minority class after splitting data).
* **Sentiment Analysis Dataset:** Text cleaning (e.g., removing punctuation, lowercasing), stop-word removal (initially considered but potentially kept for LLM), analysis using TF-IDF and n-grams, and importantly, filtering the dataset by removing classes like 'Suicidal' (due to LLM safeguards) and 'Personality Disorder'/'Bipolar' (due to high similarity with 'Depression'/'Normal' and smaller sample sizes, potentially confusing the model). The final dataset focused on 'Normal', 'Anxiety', and 'Depression' classes. Balancing the dataset was performed before training.

### 3. Exploratory Data Analysis (EDA) / Visualization (20%)

*(Summarize the EDA process and key findings)*

* **Depression Dataset:**
    * Analyzed data types and distributions.
    * Calculated correlations between features and the 'Depression' target using point-biserial correlation (for numeric/categorical vs. binary) and phi-squared coefficient (for binary vs. binary). Found no single strong predictor, suggesting a complex multi-factor relationship.
    * Performed linearity analysis using regression plots (with `logistic=True`) to visualize relationships and assess suitability for models like logistic regression, noting various relationship shapes (flat, linear, sigmoidal).
* **Anxiety Dataset:**
    * Analyzed feature correlations with the 'Anxiety' target. Identified 'Isolation', 'Future Insecurity', and 'Social Relationship' (negative) as top correlates (excluding 'Depression' itself).
    * Visualized data distributions to determine the threshold for reclassifying anxiety levels and assess class balance.
* **Sentiment Analysis Dataset:**
    * Visualized word frequencies and distributions across different mental health statuses using word clouds and frequency plots. Identified many common, non-discriminatory words initially.
    * Applied TF-IDF (unigrams and n-grams) to identify more distinctive words/phrases for each status. Noted significant overlap between certain classes.
    * Used cosine similarity on TF-IDF vectors and log-likelihood analysis to quantify the similarity/distinctiveness between classes, informing the decision to remove certain status labels.

### 4. Use of Machine Learning Techniques (20%)

*(Detail the models chosen and the rationale)*

* **Depression Prediction (ANN):** An Artificial Neural Network (ANN) was selected for the large (27,901 samples) Depression dataset. Rationale: ANN's ability to capture complex, non-linear interactions between multiple features, which is suitable given the EDA finding that individual features had low correlation but might be predictive in combination. Used ReLU activation in hidden layers and Sigmoid in the output layer (for binary probability), with Binary Cross-Entropy loss and Adam optimizer.
* **Anxiety Prediction (SVM):** Support Vector Machine (SVM) with an RBF kernel was chosen for the small (86 samples) Anxiety dataset. Rationale: SVMs can perform well on smaller datasets and find non-linear boundaries (with RBF kernel). It outperformed other tested models (Logistic Regression, Random Forest, XGBoost) on the validation set based on performance and resistance to overfitting. Feature scaling and handling class imbalance were crucial pre-processing steps.
* **Text-Based Status Prediction (Fine-tuned LLM):** Fine-tuning a pre-trained Large Language Model (Llama 3.1 8B, 4-bit quantized) was chosen for the sentiment analysis task. Rationale: To leverage the LLM's deep understanding of language, context, and semantics, going beyond simple keyword frequency (TF-IDF limitations). Fine-tuning adapts the model's general knowledge to the specific task of classifying text into 'Normal', 'Anxiety', or 'Depression'. This approach was selected over training from scratch due to significantly lower data/compute requirements and leveraging proven architectures. The model showed significant accuracy improvement post-fine-tuning (from ~73% pre-tuning to ~95.5% post-tuning).

### 5. Data-Driven Insights and Recommendations (20%)

*(Summarize key insights and the final output system)*

* **Insights:** EDA revealed the multi-factorial nature of depression and anxiety, with no single dominant predictor in the structured data. Text analysis highlighted significant linguistic overlap between different mental health statuses, motivating the use of advanced NLP models (LLMs) capable of understanding deeper semantics. Specific word patterns identified via TF-IDF and n-grams provided initial insights but confirmed the need for more contextual understanding.
* **Model Performance:** The ANN achieved [mention accuracy/performance metric] on the depression task. The SVM achieved [mention accuracy/performance metric] on the anxiety task. The fine-tuned LLM achieved ~95.5% average accuracy on the text classification task.
* **Output Fusion/Recommendations:** The final "MindScope" system integrates the predictions from the three separate models.
    1.  The ANN outputs a likelihood score (0-1) for depression based on structured data.
    2.  The SVM outputs a likelihood score (0-1) for anxiety based on structured data.
    3.  The fine-tuned LLM predicts a class ('Normal', 'Anxiety', 'Depression') based on text data.
    4.  A **Fusion Layer** combines these outputs: The LLM prediction acts as a 'nudge' (multiplier > 1 if Anxiety/Depression predicted, < 1 if Normal predicted) applied to the respective likelihood scores from the ANN/SVM. The final outputs are scaled (MinMax) to remain within the 0-1 range, representing the final predicted likelihoods for depression and anxiety.

### 6. Quality of Final Team Presentation and Overall Impressions (10%)

*(This section is for instructor evaluation based on your presentation delivery.)*
* Our presentation aimed for clarity, logical flow, and effective visualization of our process and results. We sought to demonstrate a clear understanding of the problem, methods, and outcomes.

~ insert link to presenetation slides

### 7. Learning Something New and Doing Something Beyond This Course (10%)

*(Highlight specific new techniques/concepts learned and applied)*

* **Correlation Techniques:** Learned and applied specific correlation methods suitable for mixed data types: Point-Biserial correlation (numeric/categorical vs. binary) and Phi-Squared coefficient (binary vs. binary).
* **ANN Implementation:** Gained practical experience in designing an ANN, including selecting appropriate activation functions (ReLU, Sigmoid) and loss functions (Binary Cross-Entropy) for a specific classification task.
* **Data Handling:** Practiced techniques for handling imbalanced datasets (upsampling) and the importance of feature scaling for distance-based algorithms like SVM and ANNs.
* **Advanced NLP:** Went beyond basic NLP techniques (like TF-IDF) by learning about and implementing fine-tuning for a state-of-the-art Large Language Model (Llama 3.1 8B) using quantization (4-bit) for efficient training. Explored the architectural differences between modern LLMs and the original Transformer.
* **Model Fusion:** Designed a simple fusion mechanism to combine outputs from multiple models working on different data modalities.

---
