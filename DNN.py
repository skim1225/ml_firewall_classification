import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

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

# train + predict w all features
input_shape = x_train.shape[1]
batch_size = 32
epochs = 50
num_classes = y_train.shape[1]

model_all = models.Sequential()
model_all.add(layers.Dense(20, activation='relu', input_shape=(input_shape,)))
model_all.add(layers.Dense(20, activation='relu'))
model_all.add(layers.Dense(num_classes, activation='softmax'))

model_all.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history_all = model_all.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size)

y_pred = model_all.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_back = encoder.inverse_transform(np.eye(num_classes)[y_pred_classes])

y_test_classes = np.argmax(y_test, axis=1)
y_test_back = encoder.inverse_transform(np.eye(num_classes)[y_test_classes])

accuracy_dnn_all = accuracy_score(y_test_back, y_pred_back)
precision_dnn_all = precision_score(y_test_back, y_pred_back, average='macro')
recall_dnn_all = recall_score(y_test_back, y_pred_back, average='macro')
f1_dnn_all = f1_score(y_test_back, y_pred_back, average='macro')

print("All Features - Results (DNN):")
print("Accuracy:", accuracy_dnn_all)
print("Precision:", precision_dnn_all)
print("Recall:", recall_dnn_all)
print("F1 Score:", f1_dnn_all)

# train + pred w selected features
selected_features = ['Destination Port', 'NAT Source Port', 'Source Port', 'Elapsed Time (sec)', 'Bytes Received']

x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]
input_shape = x_train_selected.shape[1]

model_selected = models.Sequential()
model_selected.add(layers.Dense(20, activation='relu', input_shape=(input_shape,)))
model_selected.add(layers.Dense(20, activation='relu'))
model_selected.add(layers.Dense(num_classes, activation='softmax'))

model_selected.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history_selected = model_selected.fit(x_train_selected,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size)

y_pred_test = model_selected.predict(x_test_selected)
y_pred_classes_test = np.argmax(y_pred_test, axis=1)
y_pred_back_test = encoder.inverse_transform(np.eye(num_classes)[y_pred_classes_test])

y_test_classes = np.argmax(y_test, axis=1)
y_test_back = encoder.inverse_transform(np.eye(num_classes)[y_test_classes])

accuracy_dnn_sel = accuracy_score(y_test_back, y_pred_back)
precision_dnn_sel = precision_score(y_test_back, y_pred_back, average='macro')
recall_dnn_sel = recall_score(y_test_back, y_pred_back, average='macro')
f1_dnn_sel = f1_score(y_test_back, y_pred_back, average='macro')

print("\nSelected Features - Results (DNN):")
print("Accuracy:", accuracy_dnn_sel)
print("Precision:", precision_dnn_sel)
print("Recall:", recall_dnn_sel)
print("F1 Score:", f1_dnn_sel)

# plot results

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
selected_metrics = [accuracy_dnn_sel, precision_dnn_sel, recall_dnn_sel, f1_dnn_sel]
all_metrics = [accuracy_dnn_all, precision_dnn_all, recall_dnn_all, f1_dnn_all]
r1 = np.arange(len(metric_names)) * 2
r2 = [x + 0.5 + 0.1 for x in r1]
metric_names_center = [(r1[i] + r2[i]) / 2 for i in range(len(r1))]
plt.bar(r1, all_metrics, width=0.5, label='All Features')
plt.bar(r2, selected_metrics, width=0.5, label='Selected Features')
plt.xlabel('Performance Metrics')
plt.ylabel('Avg Score')
plt.title('DNN')
plt.xticks(metric_names_center, metric_names)
plt.legend()
plt.show()