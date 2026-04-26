# 📊 Telecom Customer Retention AI
An end-to-end Machine Learning solution to predict customer churn with **79.6% accuracy**.

## 🚀 Project Overview
This project identifies which customers are likely to leave a telecommunications service provider. By predicting churn, businesses can intervene with targeted retention offers to save revenue.

## 🧠 Key Business Insights
After analyzing the data and training the model, the following factors were found to be the biggest drivers of churn:
1. **Total Charges:** High long-term costs correlate with higher churn risk.
2. **Tenure:** New customers are significantly more likely to leave than long-term subscribers.
3. **Contract Type:** Month-to-month customers are the highest flight risk.

## 📈 Visual Results
### 1. Top 10 Drivers of Customer Churn
This chart shows exactly what the AI looks for when predicting if a customer will leave.
![Top 10 Drivers](Top%2010%20Drivers%20of%20Customer%20Churn.png)

### 2. Confusion Matrix
A visual representation of the model's performance on unseen test data.
![Confusion Matrix](Confusion%20Matrix.png)

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Model:** Random Forest Classifier (100 Estimators)
* **Deployment:** Model serialized using `joblib` for real-time inference.

## 📂 Repository Structure
* `Customer_Retention_AI.ipynb`: Full data analysis and modeling code.
* `churn_model.pkl`: The trained "Brain" of the AI.
* `model_columns.pkl`: The feature map for the model.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used.

## 💻 How to Use
To use the trained model for your own predictions:
1. Load the model: `model = joblib.load('churn_model.pkl')`
2. Prepare your data to match the 46 features in `model_columns.pkl`.
3. Run `model.predict(new_data)`.
