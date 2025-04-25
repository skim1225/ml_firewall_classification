import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# load raw data
df = pd.read_csv('raw_if_data.csv')

# clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# separate label from features (action)
y = df['Action'].values
y = y.reshape(-1, 1)
x_data = df.drop('Action', axis=1)

# normalize features
scaler = MinMaxScaler()
x_data_norm = scaler.fit_transform(x_data)
x = pd.DataFrame(x_data_norm, columns=x_data.columns)

# perform 1-hot encoding on labels
encoder = OneHotEncoder()
encoded_y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# partition data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, test_size=0.2, random_state=42)

# examine class distribution in training dataset
y_train_labels = encoder.inverse_transform(y_train)
unique_elements, counts_elements = np.unique(y_train_labels, return_counts=True)
y_train_metrics = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
y_train_metrics.columns = ['Actions', 'Count']
print(y_train_metrics)

# examine class distribution in testing dataset
y_test_labels = encoder.inverse_transform(y_test)
unique_elements, counts_elements = np.unique(y_test_labels, return_counts=True)
y_test_metrics = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
y_test_metrics.columns = ['Actions', 'Count']
print(y_test_metrics)

# train and test SVM classifier with cleaned data

# Partition data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
selected_features = ['Destination Port', 'NAT Source Port', 'Source Port', 'Elapsed Time (sec)', 'Bytes Received']

x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]

# Train and predict SVM with selected features
svm_selected = SVC(C=1, gamma=1, kernel='rbf')
svm_selected.fit(x_train_selected, y_train)
y_pred_svm_selected = svm_selected.predict(x_test_selected)

accuracy_svm_selected = accuracy_score(y_test, y_pred_svm_selected)
precision_svm_selected = precision_score(y_test, y_pred_svm_selected, average='macro')
recall_svm_selected = recall_score(y_test, y_pred_svm_selected, average='macro')
f1_score_svm_selected = f1_score(y_test, y_pred_svm_selected, average='macro')

# Train and predict SVM with all features
svm_all = SVC(C=1, gamma=1, kernel='rbf')
svm_all.fit(x_train, y_train)
y_pred_svm_all = svm_all.predict(x_test)

accuracy_svm_all = accuracy_score(y_test, y_pred_svm_all)
precision_svm_all = precision_score(y_test, y_pred_svm_all, average='macro')
recall_svm_all = recall_score(y_test, y_pred_svm_all, average='macro')
f1_score_svm_all = f1_score(y_test, y_pred_svm_all, average='macro')

# Display results
print("All Features - Results (SVM):")
print("Accuracy:", accuracy_svm_selected)
print("Precision:", precision_svm_selected)
print("Recall:", recall_svm_selected)
print("F1 Score:", f1_score_svm_selected)

print("\nSelected Features - Results (SVM):")
print("Accuracy:", accuracy_svm_all)
print("Precision:", precision_svm_all)
print("Recall:", recall_svm_all)
print("F1 Score:", f1_score_svm_all)

# Plot data
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
selected_metrics = [accuracy_svm_selected, precision_svm_selected, recall_svm_selected, f1_score_svm_selected]
all_metrics = [accuracy_svm_all, precision_svm_all, recall_svm_all, f1_score_svm_all]

r1 = np.arange(len(metric_names)) * 2
r2 = [x + 0.5 + 0.1 for x in r1]
metric_names_center = [(r1[i] + r2[i]) / 2 for i in range(len(r1))]
plt.bar(r1, all_metrics, width=0.5, label='All Features')
plt.bar(r2, selected_metrics, width=0.5, label='Selected Features')
plt.xlabel('Performance Metrics')
plt.ylabel('Avg Score')
plt.title('SVM')
plt.xticks(metric_names_center, metric_names)
plt.legend()
plt.show()