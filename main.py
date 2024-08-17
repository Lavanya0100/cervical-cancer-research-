import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('C:\\Users\\lavgu\\Downloads\\cervical-cancer_csv.csv')

# Data preprocessing
df = df.replace('?', np.NaN)
df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], inplace=True, axis=1)

numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)']

categorical_df = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
                  'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN',
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology', 'Biopsy']

# Fill missing values
for feature in numerical_df:
    feature_mean = round(df[feature].apply(pd.to_numeric, errors='coerce').mean(), 1)
    df[feature] = df[feature].fillna(feature_mean)

for feature in categorical_df:
    df[feature] = df[feature].apply(pd.to_numeric, errors='coerce').fillna(1.0)

X = df.drop('Biopsy', axis=1).apply(pd.to_numeric, errors='coerce').astype('float64')
y = df["Biopsy"]

# Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y.ravel())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021, stratify=y)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to plot real vs predicted values
def plot_real_pred_val(Y_test, ypred, name):
    plt.figure(figsize=(25, 15))
    acc = accuracy_score(Y_test, ypred)
    plt.scatter(range(len(ypred)), ypred, color="blue", lw=5, label="Predicted")
    plt.scatter(range(len(Y_test)), Y_test, color="red", label="Actual")
    plt.title("Predicted Values vs True Values of " + name, fontsize=30)
    plt.xlabel("Accuracy: " + str(round((acc * 100), 3)) + "%", fontsize=25)
    plt.legend(fontsize=25)
    plt.grid(True, alpha=0.75, lw=1, ls='-.')
    plt.show()

# ANN Model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=500, input_dim=33, kernel_initializer='uniform', activation='relu'))
ann.add(tf.keras.layers.Dropout(0.5))
ann.add(tf.keras.layers.Dense(units=200, kernel_initializer='uniform', activation='relu'))
ann.add(tf.keras.layers.Dropout(0.5))
ann.add(tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_ann = ann.fit(X_train, y_train, batch_size=64, validation_split=0.20, epochs=10, shuffle=True)
ann.save('cervical_ann_model.h5')

# Plot accuracy and loss for ANN
def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(figsize=(25, 15))
    plt.plot(epochs, acc, 'r', label='Training accuracy', lw=10)
    plt.plot(epochs, val_acc, 'b--', label='Validation accuracy', lw=10)
    plt.title(f'Training and validation accuracy ({model_name})', fontsize=35)
    plt.legend(fontsize=25)
    ax.set_xlabel("Epoch", fontsize=30)
    ax.tick_params(labelsize=30)
    plt.show()

    fig, ax = plt.subplots(figsize=(25, 15))
    plt.plot(epochs, loss, 'r', label='Training loss', lw=10)
    plt.plot(epochs, val_loss, 'b--', label='Validation loss', lw=10)
    plt.title(f'Training and validation loss ({model_name})', fontsize=35)
    plt.legend(fontsize=25)
    ax.set_xlabel("Epoch", fontsize=30)
    ax.tick_params(labelsize=30)
    plt.show()

plot_history(history_ann, 'ANN')

# Predictions and evaluation for ANN
y_pred_ann = ann.predict(X_test)
y_pred_ann = [int(p >= 0.5) for p in y_pred_ann]
print('Accuracy Score (ANN): ', accuracy_score(y_pred_ann, y_test), '\n')
print('Classification Report (ANN):\n', classification_report(y_pred_ann, y_test))
conf_mat_ann = confusion_matrix(y_true=y_test, y_pred=y_pred_ann)
class_list = ['Biopsy = 0', 'Biopsy = 1']
fig, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(conf_mat_ann, annot=True, ax=ax, cmap='YlOrBr', fmt='g', annot_kws={"size": 25})
ax.set_xlabel('Predicted labels', fontsize=30)
ax.set_ylabel('True labels', fontsize=30)
ax.set_title('Confusion Matrix (ANN)', fontsize=30)
ax.xaxis.set_ticklabels(class_list)
ax.yaxis.set_ticklabels(class_list)
plt.show()
plot_real_pred_val(y_test, y_pred_ann, 'ANN')

# LSTM Model
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_lstm = lstm_model.fit(X_train_lstm, y_train, batch_size=64, validation_split=0.20, epochs=10, shuffle=True)
lstm_model.save('cervical_lstm_model.h5')
plot_history(history_lstm, 'LSTM')

# Predictions and evaluation for LSTM
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = [int(p >= 0.5) for p in y_pred_lstm]
print('Accuracy Score (LSTM): ', accuracy_score(y_pred_lstm, y_test), '\n')
print('Classification Report (LSTM):\n', classification_report(y_pred_lstm, y_test))
conf_mat_lstm = confusion_matrix(y_true=y_test, y_pred=y_pred_lstm)
fig, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(conf_mat_lstm, annot=True, ax=ax, cmap='YlOrBr', fmt='g', annot_kws={"size": 25})
ax.set_xlabel('Predicted labels', fontsize=30)
ax.set_ylabel('True labels', fontsize=30)
ax.set_title('Confusion Matrix (LSTM)', fontsize=30)
ax.xaxis.set_ticklabels(class_list)
ax.yaxis.set_ticklabels(class_list)
plt.show()
plot_real_pred_val(y_test, y_pred_lstm, 'LSTM')

# DBN Model (using ANN as DBN)
dbn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=500, input_dim=33, kernel_initializer='uniform', activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=200, kernel_initializer='uniform', activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
])
dbn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_dbn = dbn_model.fit(X_train, y_train, batch_size=64, validation_split=0.20, epochs=10, shuffle=True)
dbn_model.save('cervical_dbn_model.h5')
plot_history(history_dbn, 'DBN')

# Predictions and evaluation for DBN
y_pred_dbn = dbn_model.predict(X_test)
y_pred_dbn = [int(p >= 0.5) for p in y_pred_dbn]
print('Accuracy Score (DBN): ', accuracy_score(y_pred_dbn, y_test), '\n')
print('Classification Report (DBN):\n', classification_report(y_pred_dbn, y_test))
conf_mat_dbn = confusion_matrix(y_true=y_test, y_pred=y_pred_dbn)
fig, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(conf_mat_dbn, annot=True, ax=ax, cmap='YlOrBr', fmt='g', annot_kws={"size": 25})
ax.set_xlabel('Predicted labels', fontsize=30)
ax.set_ylabel('True labels', fontsize=30)
ax.set_title('Confusion Matrix (DBN)', fontsize=30)
ax.xaxis.set_ticklabels(class_list)
ax.yaxis.set_ticklabels(class_list)
plt.show()
plot_real_pred_val(y_test, y_pred_dbn, 'DBN')
