# 🚗 Car Price Prediction

This project predicts car prices based on various features using machine learning techniques such as Linear Regression. It's a beginner-friendly project for Data Science learners.

---

## 📌 Problem Statement

Estimate the price of a used car based on features like year, present price, number of owners, fuel type, seller type, and transmission.

---

## 🧠 Algorithms Used

- Linear Regression
- Random Forest (optional advanced version)

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## 📊 Dataset

The dataset used is `car_data.csv`, which contains the following columns:

- `Car_Name`
- `Year`
- `Selling_Price`
- `Present_Price`
- `Kms_Driven`
- `Fuel_Type`
- `Seller_Type`
- `Transmission`
- `Owner`

---

## 🔍 EDA (Exploratory Data Analysis)

- Removed car names for modeling.
- Converted categorical variables (Fuel Type, Seller Type, Transmission) to dummy variables.
- Created a new column: `Car_Age = Current Year - Year`.

---

## 🧪 Model Building

### python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
pred = model.predict(X_test)
r2_score(y_test, pred)


# ✅ Results
R² Score: ~0.85

The model works well for price estimation but can be improved using ensemble methods.

# 📦 Installation
bash
Copy
Edit
git clone (https://github.com/naveenk-DS/Car_Price_prediction)
cd car-price-prediction
pip install -r requirements.txt
# 💡 Future Improvements
Add more models (Random Forest, XGBoost).

Deploy using Flask or Streamlit.

Use a larger dataset for better accuracy.

# 🤝 Contributing
Feel free to fork this repo, improve the model or add a UI, and submit a pull request!

# 📧 Contact
Created by Naveen
📧 Email: naveends6k@gmail.com

---

### ✅ Sample `requirements.txt`

### txt
numpy
pandas
scikit-learn
matplotlib
seaborn
