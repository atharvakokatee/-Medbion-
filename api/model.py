import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
    roc_curve, f1_score, classification_report, confusion_matrix)
import joblib

df = pd.read_csv('data/diabetes.csv')
X = df.iloc[:,0:8]
y = df.iloc[:,8]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3,
                                max_features=0.5, random_state=0)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

print("===================================")
print("        PERFORMANCE_METRICS        ")
print("===================================")
print("ROC_AUC_SCORE:",round(roc_auc_score(y_test, y_pred_proba), 5))
print("ACCURACY_SCORE:",round(accuracy_score(y_test, y_pred), 5))
print("F1_SCORE:",round(f1_score(y_test, y_pred), 5))

# Solve the model
filename = 'finalized_model.pkl'
joblib.dump(model, filename)
joblib.dump(sc_X, 'scaler.pkl')