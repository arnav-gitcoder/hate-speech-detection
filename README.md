# 🚨 Hate Speech Detection System

A Deep Learning and Machine Learning based text classification system that automatically detects **Hate Speech**, **Offensive Language**, and **Neutral Content** from social media text.

The project implements a **Bidirectional LSTM (BiLSTM)** neural network and compares its performance against traditional machine learning models including **Logistic Regression** and **XGBoost**.

---

> **📌 Recommended Environment**
>
> This project is designed to run in **Google Colab** and **Jupyter Notebook** environments.
>
> - **Google Colab (Recommended):** No local setup required, free GPU support, and all dependencies can be installed automatically.
> - **Jupyter Notebook:** Run locally after installing the required Python packages listed in the Dependencies section.
>
> For the best experience, open `Hate_Speech_Detection.ipynb` in **Google Colab** and execute all cells sequentially from top to bottom.

---

## 🎯 Overview

Online platforms generate massive amounts of user content daily, making automated moderation increasingly important. This project explores Natural Language Processing (NLP) techniques to classify text into three categories:

| Label          | Description                                                       |
| -------------- | ----------------------------------------------------------------- |
| 🚫 Hate Speech | Targeted, harmful language directed toward an individual or group |
| ⚠️ Offensive   | Offensive or abusive language without explicit hate content       |
| ✅ Neither      | Neutral or safe content                                           |

---

## 🗂️ Project Structure

```text
hate-speech-detection/
│
├── Hate_Speech_Detection.ipynb
├── hate_speech_detection_fixed.py
├── Dataset---Hate-Speech-Detection-using-Deep-Learning.csv
│
├── hate_speech_model.keras
├── tokenizer.pkl
│
├── xgb_model.pkl
├── vectorizer.pkl
│
├── README.md
└── ML_Project_Report.pdf
```

---

## 🛠️ Tech Stack

### Languages & Frameworks

* Python
* TensorFlow / Keras
* Scikit-learn
* XGBoost

### NLP

* NLTK
* Tokenization
* Lemmatization
* Stop-word Removal
* TF-IDF Vectorization

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Deployment

* Gradio

---

## 🤖 Models Implemented

### 1. Bidirectional LSTM (BiLSTM)

Deep Learning model designed for sequential text understanding.

**Architecture**

```text
Embedding Layer
        ↓
Bidirectional LSTM (64)
        ↓
Dropout (0.3)
        ↓
Bidirectional LSTM (32)
        ↓
Dropout (0.3)
        ↓
Dense Softmax Layer (3 Classes)
```

**Training Features**

* EarlyStopping
* ReduceLROnPlateau
* Adam Optimizer
* Sparse Categorical Crossentropy

---

### 2. Logistic Regression

Traditional machine learning baseline using:

* TF-IDF Features
* Bigrams (1,2)
* Maximum 10,000 Features

Advantages:

* Fast Training
* Lightweight
* Easily Interpretable

---

### 3. XGBoost

Gradient Boosting based classifier using:

* TF-IDF Features
* Multi-class Classification
* Histogram-based Training

Advantages:

* Strong Performance
* Handles Complex Feature Interactions
* Efficient Prediction

---

## 📊 Project Workflow

| Step | Description                   |
| ---- | ----------------------------- |
| 1    | Install Dependencies          |
| 2    | Import Libraries              |
| 3    | Load Dataset                  |
| 4    | Data Cleaning & Preprocessing |
| 5    | Train/Test Split (80/20)      |
| 6    | Tokenization                  |
| 7    | Sequence Padding              |
| 8    | Build BiLSTM Model            |
| 9    | Train Model                   |
| 10   | Model Evaluation              |
| 11   | Save Model & Tokenizer        |
| 12   | Prediction Pipeline           |
| 13   | Logistic Regression Baseline  |
| 14   | XGBoost Model                 |
| 15   | Model Comparison              |
| 16   | Gradio Deployment             |

---

## 🧹 Text Preprocessing Pipeline

The following preprocessing steps are applied before training:

* Convert text to lowercase
* Remove punctuation
* Remove stopwords
* Lemmatize words
* Tokenize text
* Convert to padded sequences

Example:

```text
Input:
"I HATE people like you!!!"

Output:
"hate people like"
```

---

## 📦 Dataset

### Hate Speech and Offensive Language Dataset

* Source: Davidson et al.
* Size: ~24,000 Tweets
* Classes: 3

| Class | Meaning            |
| ----- | ------------------ |
| 0     | Hate Speech        |
| 1     | Offensive Language |
| 2     | Neither            |

### Class Balancing

To address class imbalance, minority classes were resampled to create a more balanced training distribution.

---

## 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/arnav-gitcoder/hate-speech-detection.git
cd hate-speech-detection
```

### Install Dependencies

```bash
pip install tensorflow scikit-learn xgboost nltk wordcloud gradio pandas numpy matplotlib seaborn
```

### Run the Project

```bash
python hate_speech_detection_fixed.py
```

Or open:

```text
Hate_Speech_Detection.ipynb
```

in Google Colab and run all cells sequentially.

---

## 🖥️ Gradio Interface

The project includes a Gradio-based web interface for real-time text classification.

### Example

```text
Input:
"I hate people like you"

BiLSTM              → 🚫 Hate Speech
Logistic Regression → ⚠️ Offensive
XGBoost             → 🚫 Hate Speech
```

---

## 📈 Model Evaluation

The project evaluates:

* Accuracy
* Validation Loss
* Classification Report
* Precision
* Recall
* F1 Score

Model comparison enables analysis of the trade-off between:

* Accuracy
* Training Time
* Inference Speed
* Model Complexity

---

## 🔮 Future Improvements

* BERT / RoBERTa based models
* Explainable AI (SHAP/LIME)
* REST API deployment
* Docker containerization
* Real-time social media monitoring
* Cloud deployment using AWS or Azure

---

## 👨‍💻 Authors

**Arnav Srivastava**
GitHub: https://github.com/arnav-gitcoder

**Ayush Kaushik**

---

## ⭐ If you found this project useful, consider giving the repository a star!
