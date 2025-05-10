import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Dummy fintech dataset
data = pd.DataFrame({
    'transaction_amount': [20, 250, 13, 450, 70, 5, 999, 120, 35, 310],
    'account_age_days': [365, 200, 30, 700, 100, 5, 1200, 80, 45, 600],
    'payment_method': [1, 2, 0, 1, 2, 0, 1, 1, 2, 0],  # 0=debit, 1=credit, 2=ACH
    'is_fraud_suspected': [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    'failed_payment': [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
})

X = data.drop('failed_payment', axis=1)
y = data['failed_payment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Fintech Payment Failure Prediction Report:")
print(classification_report(y_test, preds))
