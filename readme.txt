# 🏠 House Rent Price Prediction using Machine Learning

## 📌 Overview

This project predicts house rental prices using Machine Learning based on property characteristics such as:

- Property Size
- Area (sqft)
- Property Type
- Location
- Seller Type
- Furnishing Status
- Bathrooms

The objective is to estimate rental prices and identify the factors influencing rent.

---

## 📊 Dataset

- Total Records: 14,000
- Features: 13
- Target Variable:
  - Rent_price

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## ⚙ Data Preprocessing

- Removed unnecessary columns
- Converted rent values into numeric format
- Handled missing values
- One-Hot Encoding for categorical variables
- Train-Test Split

---

## 🤖 Machine Learning Model

Random Forest Regressor

---

## 📈 Model Performance

- R² Score: **0.8775**
- MAE: **42892**
- RMSE: **97001**

---

## 📊 Visualizations

- Actual vs Predicted Rent
- Feature Importance Analysis

---

## 🔍 Key Findings

- Area (sqft) is the most influential feature.
- Premium locations significantly increase rent.
- Property type also strongly impacts rental prices.

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Better handling of location features
- Deployment using Streamlit or Flask

---

## 👨‍💻 Author

Mohd Labeeb Zaidi