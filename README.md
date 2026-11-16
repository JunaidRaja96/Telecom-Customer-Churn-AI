# 📊 Telecom Customer Retention AI
A Random Forest model that predicts which telecom customers are likely to churn. Test accuracy is **79.6%**, and with a tuned decision threshold it catches **77% of churners** (recall) instead of 48%.

## 🚀 Project Overview
This project identifies which customers are likely to leave a telecommunications service provider. By predicting churn, businesses can intervene with targeted retention offers to save revenue.

- **Data:** Telco Customer Churn dataset, 7,043 customers, 26.5% churned
- **Model:** Random Forest (100 trees), 46 features after one-hot encoding
- **Split:** 80% train / 20% test (1,409 test customers), `random_state=42`

## 🧠 Key Business Insights
Churn rates computed directly from the dataset:

| Segment | Churn rate |
|---|---|
| Month-to-month contract | **42.7%** |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Tenure 0-12 months | **47.4%** |
| Tenure 13-24 months | 28.7% |
| Tenure 25+ months | 14.0% |

1. **Contract type:** month-to-month customers are by far the highest flight risk.
2. **Tenure:** new customers are much more likely to leave than long-term subscribers.
3. **Monthly charges:** churners pay more per month on average (74.4 vs 61.3).
4. **Total charges** is the top feature by importance, but churners actually have *lower* total charges (median 704 vs 1,684). This is because total charges mostly reflects how long someone has been a customer (tenure × monthly charge).

Note: feature importance shows how much the model *uses* a feature, not the direction of the effect. The table above shows the direction.

## 📈 Visual Results
### 1. Top 10 Drivers of Customer Churn (Random Forest feature importance)
![Top 10 Drivers](Top%2010%20Drivers%20of%20Customer%20Churn.png)

### 2. Confusion Matrix (default 0.5 threshold)
A visual representation of the model's performance on unseen test data.
![Confusion Matrix](Confusion%20Matrix.png)

## 🎯 Threshold Tuning
Churn is the minority class, so accuracy alone is misleading: a model that always predicts "no churn" would already score 73.5%. At the default 0.5 threshold, recall on churners was only 0.48.

Lowering the threshold trades precision for recall:

| Threshold | Accuracy | Precision | Recall |
|---|---|---|---|
| 0.5 (default) | 0.796 | 0.65 | 0.48 |
| 0.4 | 0.789 | 0.59 | 0.64 |
| 0.3 | 0.759 | 0.53 | 0.77 |

For a retention campaign, missing a churner usually costs more than sending an offer to someone who would have stayed, so a lower threshold is the better trade-off. The right value depends on the cost of an offer vs the value of a retained customer.

```python
probs = model.predict_proba(X_test)[:, 1]
pred = (probs >= 0.3).astype(int)
```

## ⚠️ Limitations
- Single train/test split (no cross-validation yet)
- No hyperparameter tuning; `class_weight` and other imbalance methods not yet compared
- Threshold was chosen by looking at the test set; a separate validation set would be cleaner

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Model:** Random Forest Classifier (100 Estimators)
* **Serialization:** Model saved with `joblib` for reuse and inference.

## 📂 Repository Structure
* `Customer_Retention_AI_Predicting_Subscriber_Churn.ipynb`: Full data analysis and modeling code.
* `churn_model.pkl`: The trained model.
* `model_columns.pkl`: The feature map for the model.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used.

## 💻 How to Use
To use the trained model for your own predictions:
1. Load the model: `model = joblib.load('churn_model.pkl')`
2. Prepare your data to match the 46 features in `model_columns.pkl`.
3. Get churn probabilities and apply your threshold: `model.predict_proba(new_data)[:, 1] >= 0.3`
