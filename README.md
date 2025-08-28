#  Credit Card Fraud Detection System

A **real-time credit card fraud detection** system built with **Python**, **Scikit-learn**, and **Streamlit**.  
This project demonstrates how machine learning can detect fraudulent transactions from a credit card dataset.

---

##  Project Description

Banks lose millions every year due to fraudulent transactions. This project aims to **detect fraud in real-time** using a trained ML model.  

Key features:  
- Detect fraud using **Random Forest** (or other ML models)  
- Interactive **Streamlit app** for predictions  
- **EDA (Exploratory Data Analysis)** with visualizations  
- Handles **imbalanced datasets** using techniques like SMOTE  
- Shows **fraud probability** for each transaction  
- Users can test using **random example transactions** or manually input features  

The dataset contains anonymized features (`V1`–`V28`) derived from PCA for privacy reasons, along with `Time` and `Amount` features.  

---

##  Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Features:
  - `Time`: Seconds elapsed between transaction and first transaction
  - `Amount`: Transaction amount
  - `V1`–`V28`: PCA-transformed features to protect sensitive information
  - `Class`: 0 = Legit, 1 = Fraud

---

##  Model Accuracy

The project uses a **Random Forest Classifier** trained on the Kaggle credit card dataset.  

- **Evaluation Metrics:**  
  - **Precision:** ~0.92  
  - **Recall:** ~0.86  
  - **F1-Score:** ~0.89  
  - **Accuracy:** ~0.999  

---

##  Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/CreditCard-Fraud-Detection.git
cd CreditCard-Fraud-Detection
