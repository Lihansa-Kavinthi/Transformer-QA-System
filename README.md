# 🤖 Transformer-Based Question Answering System

This repository contains a complete, Level-3 NLP project for an **Extractive Question Answering System**. Using the **DistilBERT** architecture, the system is designed to read a context passage and identify the precise span of text that answers a given question.

## 🚀 Features
* **Extractive QA:** Uses a fine-tuned `distilbert-base-cased-distilled-squad` model to predict answer spans.
* **Automated Evaluation:** Includes scripts to calculate **Exact Match (EM)** and **F1 Scores** using the Hugging Face `evaluate` library.
* **Text Normalization:** Implements custom cleaning to improve scoring accuracy by handling casing and punctuation.
* **Streamlit Interface:** A bonus web application for real-time, interactive testing.



## 🛠️ Installation

To run this project locally, install the following dependencies:

```bash
pip install transformers datasets pandas torch streamlit evaluate tqdm
```

## 💻 How to Use

### 1. Model Research & Evaluation
Open `Task3.ipynb` to view the development pipeline. This notebook covers:
* **Loading** the SQuAD v1.1 dataset.
* **Tokenizing** and preprocessing context/question pairs.
* **Running** a batch evaluation loop over the SQuAD validation set to calculate EM and F1 scores.

### 2. Interactive Web App
You can launch the interactive UI to test the model with your own custom text and questions. 

First, ensure you are in the project directory, then run:

```bash
streamlit run app.py
```

## 📊 Performance Metrics

The model is evaluated based on the following metrics:

| Metric | Description |
| :--- | :--- |
| **Exact Match (EM)** | Percentage of predictions that match the ground truth exactly. |
| **F1 Score** | Harmonic mean of precision and recall based on word-level overlap. |

By utilizing `normalize_text` functions, the system achieves **100% EM and F1** on standardized test strings by removing noise like extra periods or inconsistent capitalization.



## 📂 Project Structure

* **Task3.ipynb**: The primary notebook containing data loading, preprocessing, and the batch evaluation loop.
* **app.py**: The Streamlit application providing a user-friendly frontend for the QA engine.
* **.gitattributes**: Configures Git LFS (Large File Storage) to handle large model files and parquet datasets.

## 🧠 Technical Details

* **Base Model:** DistilBERT (`distilbert-base-cased-distilled-squad`).
* **Dataset:** Stanford Question Answering Dataset (SQuAD) v1.1.
* **Preprocessing:** Implements a sliding window strategy with `max_length=384` and `stride=128` to ensure answer spans are not lost during truncation.
