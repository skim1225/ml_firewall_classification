import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and clean data
df = pd.read_csv("raw_if_data.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Separate features and label
X = df.drop('Action', axis=1)
y = df['Action']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# train/test on all features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_rf_all = accuracy_score(y_test, y_pred)
precision_rf_all = precision_score(y_test, y_pred, average='macro')
recall_rf_all = recall_score(y_test, y_pred, average='macro')
f1_rf_all = f1_score(y_test, y_pred, average='macro')

print("All Features - Results (Random Forest):")
print("Accuracy :", accuracy_rf_all)
print("Precision:", precision_rf_all)
print("Recall   :", recall_rf_all)
print("F1 Score :", f1_rf_all)

# rf feature selection
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)

top_5_features = feature_importance_df['Feature'].iloc[:5].tolist()
print("Selected Features:", top_5_features)

# train/test on selected features
X_df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_selected = X_df_scaled[top_5_features]

X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

rf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sel.fit(X_train_selected, y_train_selected)
y_pred_sel = rf_sel.predict(X_test_selected)

accuracy_rf_selected = accuracy_score(y_test_selected, y_pred_sel)
precision_rf_selected = precision_score(y_test_selected, y_pred_sel, average='macro')
recall_rf_selected = recall_score(y_test_selected, y_pred_sel, average='macro')
f1_rf_selected = f1_score(y_test_selected, y_pred_sel, average='macro')

print("Selected Features - Results (Random Forest):")
print("Accuracy:", accuracy_rf_selected)
print("Precision:", precision_rf_selected)
print("Recall:", recall_rf_selected)
print("F1 Score", f1_rf_selected)

# Plot data
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
selected_metrics = [accuracy_rf_selected, precision_rf_selected, recall_rf_selected, f1_rf_selected]
all_metrics = [accuracy_rf_all, precision_rf_all, recall_rf_all, f1_rf_all]
r1 = np.arange(len(metric_names)) * 2
r2 = [x + 0.5 + 0.1 for x in r1]
metric_names_center = [(r1[i] + r2[i]) / 2 for i in range(len(r1))]
plt.bar(r1, all_metrics, width=0.5, label='All Features')
plt.bar(r2, selected_metrics, width=0.5, label='Selected Features')
plt.xlabel('Performance Metrics')
plt.ylabel('Avg Score')
plt.title('Random Forest')
plt.xticks(metric_names_center, metric_names)
plt.legend()
plt.show()
