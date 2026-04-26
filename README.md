# IrisFlowerClassification
Iris Flower Classification project using machine learning to predict species (Setosa, Versicolor, Virginica) from petal and sepal measurements. Implements Logistic Regression, KNN, and Decision Tree with data visualization, preprocessing, and evaluation using accuracy, classification report, and confusion matrix.
```py
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame for better handling
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

print(df.head())

# -----------------------------
# 2. Data Visualization
# -----------------------------
sns.pairplot(df, hue='species')
plt.show()

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Feature Scaling (important for some models)
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. Train Model (Logistic Regression)
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))```


print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
