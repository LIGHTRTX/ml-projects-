# Machine Learning and AI Projects

This repository contains multiple applied projects across domains such as imaging, NLP, signal processing, finance, healthcare, and environmental data science. Each project focuses on solving a clearly defined problem using machine learning, deep learning, and statistical analysis techniques. Implementations are primarily developed in Python using Jupyter notebooks and standard ML libraries.

---

# Project Descriptions

## 1. Hyperspectral Image Approximation

**Objective:**
Approximate hyperspectral image data from standard RGB images to enable spectral analysis without requiring expensive hyperspectral sensors.

**Methods:**

* Spectral super-resolution using deep learning models such as **HSCNN+, MST++, and HyperReconNet**
* Dimensionality reduction using **Principal Component Analysis (PCA)**
* Reconstruction of spectral bands from RGB channels

**Evaluation Metrics**

* Peak Signal-to-Noise Ratio (PSNR)
* Structural Similarity Index (SSIM)
* Spectral Angle Mapper (SAM)

**Applications**

* Remote sensing
* Agricultural monitoring
* Medical imaging
* Environmental analysis

---

## 2. Text Classification and Regional NLP Analysis

**Objective:**
Perform text classification on multilingual and regional datasets, including code-mixed text such as Tanglish (Tamil-English).

**Methods**

* Text preprocessing (tokenization, stopword removal, normalization)
* Feature extraction using **TF-IDF and word embeddings**
* Traditional ML models: **Logistic Regression, Random Forest**
* Transformer-based models: **BERT, IndicBERT, XLM-R**

**Applications**

* Sentiment analysis
* Social media monitoring
* Intent detection
* Regional language NLP research

---

## 3. EEG Analysis

**Objective:**
Analyze EEG signals to detect cognitive states and neurological patterns.

**Methods**

* EEG preprocessing using **MNE-Python**
* Artifact removal and bandpass filtering
* Power spectral density analysis across brain wave bands:

  * Delta
  * Theta
  * Alpha
  * Beta
* Event Related Potential (ERP) analysis

**Applications**

* Brain-computer interfaces
* Cognitive workload monitoring
* Fatigue detection
* Medical diagnostics

---

## 4. Fake News Detection

**Objective:**
Develop machine learning models to classify news articles as real or fake.

**Methods**

* NLP preprocessing and feature engineering
* Vectorization using TF-IDF and embeddings
* Models implemented:

  * Logistic Regression
  * Random Forest
  * LSTM
  * Transformer-based classifiers

**Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1 Score

**Applications**

* Automated misinformation detection
* Content moderation
* Media monitoring systems

---

## 5. Financial Fraud Detection

**Objective:**
Detect fraudulent transactions within large-scale financial datasets.

**Methods**

* Handling class imbalance using **SMOTE and ADASYN**
* Anomaly detection techniques
* Models implemented:

  * Isolation Forest
  * One-Class SVM
  * Gradient Boosting
  * XGBoost

**Applications**

* Banking fraud detection
* Insurance claim verification
* E-commerce transaction monitoring

---

## 6. Binary Classification of Alpha vs Hydron Particles

**Objective:**
Differentiate between alpha particles and hydron particles using machine learning models.

**Methods**

* Feature extraction from particle physics datasets
* Supervised classification models including:

  * Support Vector Machines (SVM)
  * Random Forest
  * Neural Networks

**Evaluation Metrics**

* ROC-AUC
* Precision-Recall analysis
* Classification accuracy

**Applications**

* Particle detection systems
* Physics experiment data analysis
* Scientific instrumentation

---

## 7. Noisy Dataset Handling with Cleanlab and NumPy

**Objective:**
Improve machine learning model performance in the presence of noisy or mislabeled data.

**Methods**

* Label error detection using **Cleanlab**
* Dataset preprocessing using **NumPy pipelines**
* Comparative experiments between:

  * Baseline models
  * Noise-cleaned datasets

**Applications**

* Real-world datasets with annotation errors
* Robust model training
* Data quality improvement pipelines

---

## 8. Water Pollution Analysis

**Objective:**
Analyze water quality datasets to evaluate pollution levels and environmental health.

**Methods**

* Statistical analysis of environmental indicators including:

  * pH
  * Turbidity
  * Dissolved oxygen
  * Chemical concentrations
* Machine learning classification and clustering
* Visualization using heatmaps and time-series plots

**Applications**

* Environmental monitoring
* Urban water management
* Public health analysis

---

## 9. Weather Prediction for Chennai

**Objective:**
Develop forecasting models for weather prediction in Chennai using historical meteorological data.

**Methods**

* Time-series forecasting models including:

  * ARIMA
  * Facebook Prophet
  * LSTM networks
* Feature engineering from variables such as temperature, humidity, and rainfall

**Evaluation Metrics**

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

**Applications**

* Agriculture planning
* Disaster preparedness
* Smart city infrastructure

---

## 10. Speech Emotion and Voice Analysis

**Objective:**
Analyze emotional characteristics in speech signals using a large dataset of actors expressing different emotions.

**Dataset**

Ryerson Audio-Visual Database of Emotional Speech and Song (**RAVDESS**)

**Dataset Scale**

* 24 professional actors
* 70 emotional state variations
* More than 5,000 speech recordings

**Methods**

* Audio preprocessing using **Librosa**
* Feature extraction including:

  * MFCC (Mel Frequency Cepstral Coefficients)
  * Pitch (fundamental frequency)
  * RMS energy
  * Spectral centroid
  * Zero-crossing rate
* Visualization and analysis using **Matplotlib and Seaborn**
* Dimensionality reduction using **PCA and t-SNE**

**Applications**

* Speech emotion recognition
* Voice assistants
* Human-computer interaction
* Behavioral and psychological analysis

---

## 11. Heart Disease Prediction

**Objective:**
Predict the risk of heart disease using clinical patient data and machine learning models.

**Dataset**

UCI Heart Disease Dataset

**Features**

* Age
* Blood pressure
* Cholesterol
* ECG results
* Chest pain type
* Maximum heart rate
* Additional clinical indicators

**Methods**

* Data preprocessing and feature engineering
* Handling missing values and normalization
* Machine learning models:

  * Logistic Regression
  * Random Forest
  * Support Vector Machine
  * Gradient Boosting

**Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

**Applications**

* Clinical decision support systems
* Early detection of cardiovascular risk
* Healthcare analytics

---

# Technologies and Tools

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / PyTorch
* Librosa
* MNE-Python
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Skills Demonstrated

* Image super-resolution and spectral approximation
* Multilingual NLP and text classification
* EEG and biomedical signal processing
* Speech processing and emotion recognition
* Anomaly detection and fraud analytics
* Particle physics classification
* Data cleaning and noisy label detection using Cleanlab
* Environmental data analysis
* Healthcare prediction models
* Time-series forecasting
* Deep learning with CNNs, RNNs, and Transformers
