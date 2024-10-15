import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("cars data.csv")

# Features and Labels (non-numeric data to be encoded)
features = ["First Name", "Last Name", "Car Brand", "Car Model", "Car Color", "Credit Card Type"]
X = data[features]
Y = data["Year of Manufacture"]

# Encoding categorical features
le = LabelEncoder()
for col in X.columns:
    X.loc[:, col] = le.fit_transform(X[col])

# Normalizing the data for better performance of decision tree algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- Split dataset: 70%-30% for Decision Tree -------------------
X_train_70, X_test_30, Y_train_70, Y_test_30 = train_test_split(X_scaled, Y, test_size=0.3, random_state=1)

# ------------------- Decision Tree Classifier (CART) -------------------
decision_tree = DecisionTreeClassifier(max_depth=20, criterion="gini", random_state=1)
decision_tree.fit(X_train_70, Y_train_70)

# Predict the response for the test dataset (30% split)
Y_pred_dt_30 = decision_tree.predict(X_test_30)

# ------------------- Metrics for Decision Tree -------------------
accuracy_dt = accuracy_score(Y_test_30, Y_pred_dt_30)
precision_dt = precision_score(Y_test_30, Y_pred_dt_30, average='macro', zero_division=0)
recall_dt = recall_score(Y_test_30, Y_pred_dt_30, average='macro', zero_division=0)
f1_dt = f1_score(Y_test_30, Y_pred_dt_30, average='macro')

print(f"Decision Tree (70%-30%) - Accuracy: {accuracy_dt * 100:.2f}%")
print(f"Precision: {precision_dt:.2f}")
print(f"Recall: {recall_dt:.2f}")
print(f"F1 Score: {f1_dt:.2f}")

# ------------------- Confusion Matrix for Decision Tree -------------------
conf_matrix_dt = confusion_matrix(Y_test_30, Y_pred_dt_30)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Decision Tree")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ------------------- Optional: AUC for Decision Tree (if binary) -------------------
try:
    if len(set(Y)) == 2:
        auc_dt = roc_auc_score(Y_test_30, decision_tree.predict_proba(X_test_30)[:, 1])  # Binary AUC
        print(f"AUC (Decision Tree): {auc_dt:.2f}")
except ValueError:
    print("AUC is not applicable for multi-class without adjustment.")

# ------------------- Split dataset: 80%-20% for ID3 -------------------
X_train_80, X_test_20, Y_train_80, Y_test_20 = train_test_split(X_scaled, Y, test_size=0.2, random_state=1)

# ------------------- ID3 Classifier (Using entropy criterion) -------------------
id3_tree = DecisionTreeClassifier(criterion="entropy", max_depth=20, random_state=1)  # ID3 uses entropy
id3_tree.fit(X_train_80, Y_train_80)

# Predict the response for the test dataset (20% split)
Y_pred_id3_20 = id3_tree.predict(X_test_20)

# ------------------- Metrics for ID3 -------------------
accuracy_id3 = accuracy_score(Y_test_20, Y_pred_id3_20)
precision_id3 = precision_score(Y_test_20, Y_pred_id3_20, average='macro', zero_division=0)
recall_id3 = recall_score(Y_test_20, Y_pred_id3_20, average='macro', zero_division=0)
f1_id3 = f1_score(Y_test_20, Y_pred_id3_20, average='macro')

print(f"ID3 (80%-20%) - Accuracy: {accuracy_id3 * 100:.2f}%")
print(f"Precision: {precision_id3:.2f}")
print(f"Recall: {recall_id3:.2f}")
print(f"F1 Score: {f1_id3:.2f}")

# ------------------- Confusion Matrix for ID3 -------------------
conf_matrix_id3 = confusion_matrix(Y_test_20, Y_pred_id3_20)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix_id3, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for ID3")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ------------------- Optional: AUC for ID3 (if binary) -------------------
try:
    if len(set(Y)) == 2:
        auc_id3 = roc_auc_score(Y_test_20, id3_tree.predict_proba(X_test_20)[:, 1])  # Binary AUC
        print(f"AUC (ID3): {auc_id3:.2f}")
except ValueError:
    print("AUC is not applicable for multi-class without adjustment.")
