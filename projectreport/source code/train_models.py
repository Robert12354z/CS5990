from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Load training data from CSV files
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
joblib.dump(nb_model, "data/naive_bayes_model.pkl")
print("Naive Bayes model saved.")

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, "data/decision_tree_model.pkl")
print("Decision Tree model saved.")

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, "data/logistic_regression_model.pkl")
print("Logistic Regression model saved.")

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "data/random_forest_model.pkl")
print("Random Forest model saved.")
