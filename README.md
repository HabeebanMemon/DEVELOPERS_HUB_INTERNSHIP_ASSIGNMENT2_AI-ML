# DEVELOPERS_HUB_INTERNSHIP_ASSIGNMENT2

## 🔹 Tasks Overview
This repository contains internship project submissions for Developers Hub AI/ML Internship, showcasing advanced machine learning techniques including pipelines, multimodal models, and LLM applications using real-world datasets.

---

# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## 🔍 Objective
Build a reusable and production-ready machine learning pipeline to predict customer churn using the Telco Churn Dataset.

## 📂 Dataset
- Source: Telco Churn Dataset
- Features: Includes numerical (e.g., tenure, MonthlyCharges) and categorical (e.g., gender, Contract) data.

## 📊 Features Used
- **Numerical:** tenure, MonthlyCharges
- **Categorical:** gender, Contract
- **Target:** Churn (0 = No, 1 = Yes)

## 🛠️ Models Applied
- Linear Regression
- Random Forest Regressor

## 🔢 Evaluation Metrics
| Model              | Cross-Validation Score | Best Parameters          |
|-------------------|-------------------------|--------------------------|
| Logistic Regression| [insert value]          | C = [insert value]       |
| Random Forest      | [insert value]          | n_estimators = [insert value], max_depth = [insert value] |

## 📉 Visualization
[Add visualization of churn prediction accuracy or feature importance if implemented.]

## 💪 Tech Stack
- Python
- scikit-learn
- pandas, numpy
- joblib

## 🛠️ Methodology
- Data preprocessing with `Pipeline` using `StandardScaler` and `OneHotEncoder`.
- Hyperparameter tuning with `GridSearchCV`.
- Model export using `joblib` for production readiness.

## ✅ Results
- Best Logistic Regression score improved by [insert percentage]% with tuning.
- Random Forest achieved [insert value] accuracy with optimized parameters.
- Pipeline exported successfully with a file size of [insert size] MB.
[Update with actual results after running `Task2/main.py`.]

---

# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

## 🔍 Objective
Predict housing prices by combining structured data from Housing.csv with features extracted from house images.

## 📂 Dataset
- **Tabular:** Housing.csv (545 records) with features like area, bedrooms.
- **Images:** 30,948 house images from `C:\Users\Habeeban Memon\Desktop\Assignment2\socal2\socal_pics`.

## 📊 Features Used
- **Numerical (Tabular):** area, bedrooms, bathrooms, stories, parking
- **Categorical (Tabular):** mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus
- **Image:** Extracted via ResNet50 CNN
- **Target:** price

## 🧪 Steps Performed
- Preprocessed tabular data with `StandardScaler` and `OneHotEncoder`.
- Used a `tf.data.Dataset` generator for memory-efficient image loading.
- Trained a multimodal model fusing CNN and tabular features.
- Evaluated using MAE and RMSE.

## 🔢 Evaluation Metrics
| Metric       | Value    |
|--------------|----------|
| MAE          | [insert value] |
| RMSE         | [insert value] |

## 📉 Visualization
[Add plot of actual vs predicted prices if implemented.]

## 💪 Tech Stack
- Python
- tensorflow
- scikit-learn
- pandas, numpy
- matplotlib (optional)

## ✅ Results
- Achieved an MAE of [insert value] and RMSE of [insert value] on the test set.
- CNN features improved prediction by [insert percentage]% over tabular-only models.
- Training converged after [insert number] epochs with batch size 32.
[Update with actual results after running `Task3/main.py`.]

---

# Task 5: Auto Tagging Support Tickets Using LLM

## 🔍 Objective
Automatically tag support tickets into categories using a large language model (LLM), comparing zero-shot, few-shot, and fine-tuned approaches.

## 📂 Dataset
- Simulated free-text support ticket dataset with 5 samples (replace with your dataset).

## 📊 Features Used
- **Input:** ticket_text (e.g., "My internet is down, please help!")
- **Categories:** Billing, Technical, General Inquiry, Account Management
- **Target:** true_label (e.g., Technical)

## 🧪 Steps Performed
- Zero-shot classification with `facebook/bart-large-mnli`.
- Few-shot learning simulated with `distilbert-base-uncased` using example prompts.
- Fine-tuning simulation with `distilbert-base-uncased` on a small dataset.
- Evaluated with classification reports.

## 🔢 Evaluation Metrics
| Method         | F1-Score   |
|----------------|------------|
| Zero-Shot      | [insert value] |
| Few-Shot       | [insert value] |
| Fine-Tuned     | [insert value] |

## 📉 Visualization
[Add confusion matrix or tag distribution plot if implemented.]

## 💪 Tech Stack
- Python
- transformers
- torch
- pandas, numpy
- scikit-learn

## ✅ Results
- Zero-shot F1-score of [insert value], with high recall for Technical.
- Few-shot improved F1-score to [insert value] for Billing.
- Fine-tuning reached [insert value] F1-score, outperforming zero-shot by [insert percentage]%.
- Top 3 tags ranked with [insert accuracy percentage]% accuracy.
[Update with actual results after running `Task5/main.py`.]

---

## 👤 Developer Info
- **Internship**: Developers Hub - AI/ML Track Assignment_2
- **Duration**: Summer 2025
- **Developer**: [Your Name] (GitHub: `yourusername`)

## 📋 Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DEVELOPERS_HUB_INTERNSHIP_ASSIGNMENT2.git
