# 📡 Telecom Product Churn Prediction

A machine learning project aimed at predicting which customers are likely to churn from a telecom service provider. This project explores multiple classification algorithms, evaluates their performance, and identifies the best model for deployment.

---

## 🚀 Project Overview

Customer churn is a major challenge for telecom companies. Predicting churn enables proactive retention strategies that significantly reduce revenue loss. This project analyzes customer data and builds predictive models using:

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **Support Vector Machine (SVM)**

Each model is evaluated using key performance metrics:

* **Accuracy**
* **Recall Score**
* **Precision Score**
* **F1 Score**

---

## 🗂️ Dataset

* The dataset includes customer demographics, service usage patterns, billing information, and churn status.
* The target variable is: **Churn (Yes/No)**.
* Basic preprocessing steps include:

  * Handling missing values
  * Encoding categorical variables
  * Scaling numerical features (where required)

---

## 🧠 Machine Learning Workflow

### 1. **Data Cleaning & Preprocessing**

* Removed null values
* Label encoding & One-hot encoding for categorical columns
* Feature scaling using StandardScaler (for models sensitive to magnitude)

### 2. **Model Training**

Trained four classification models:

* Logistic Regression
* Decision Tree
* Random Forest
* SVM

### 3. **Model Evaluation**

Evaluated all models using:

* **Accuracy** → Overall performance
* **Precision** → How many predicted churns were correct
* **Recall** → How many actual churns were correctly identified
* **F1 Score** → Balance between precision & recall

### 4. **Model Comparison**

All four models were compared and the model with the best balance of recall & F1-score was chosen.

---

## 📊 Results Summary

*(Add your actual values here)*

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | -        | -         | -      | -        |
| Decision Tree       | -        | -         | -      | -        |
| Random Forest       | -        | -         | -      | -        |
| SVM                 | -        | -         | -      | -        |

> **Tip:** Random Forest and SVM usually perform well in churn problems, but recall is critical since detecting actual churners matters most.

---

## 📁 Project Structure

```
Telecom-Churn/
│
├── data/                # Dataset (not included on GitHub if private)
├── notebooks/           # Jupyter Notebooks for EDA & model building
├── scripts/             # Python scripts for training & evaluation
├── models/              # Saved model files (pickles)
├── README.md            # Project documentation
└── requirements.txt      # Dependencies
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/telecom-churn.git
cd telecom-churn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python churn_prediction.py
```

---

## 🔍 Exploratory Data Analysis Highlights

* Customer tenure is highly correlated with churn
* High monthly charges often indicate churn risk
* Customers using fiber-optic service appear more likely to churn
* Contract type and payment method significantly influence churn behavior

---

## 🏆 Best Performing Model

Your selected model: **(Add here e.g., Random Forest)**

Why?

* High recall → correctly identifies majority of actual churners
* Balanced precision → fewer false positives
* Strong F1 score

---

## 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV / RandomizedSearchCV
* Add feature importance visualization
* Deploy model using Flask / FastAPI
* Create dashboard for churn monitoring

---

## 🤝 Contributing

Pull requests are welcome! If you want to suggest improvements, feel free to open an issue.

---

## 📬 Contact

**Your Name:** Namit Kumar
**GitHub:** [https://github.com/your-username](https://github.com/your-username)

If you found this project helpful, don’t forget to ⭐ the repo!
