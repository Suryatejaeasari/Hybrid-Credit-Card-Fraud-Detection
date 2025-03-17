import pandas as pd
#import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score  # Import accuracy_score

# Load the dataset
df = pd.read_csv('train_transaction.csv')

# Define the features and target
features = df.drop('isFraud', axis=1)
target = df['isFraud']

# Identify categorical and numerical columns
categorical_features = features.select_dtypes(include=['object', 'category']).columns
numerical_features = features.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for both numerical and categorical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Preprocess the data
X_processed = preprocessor.fit_transform(features)

# Ensure target is aligned correctly
target = target.reset_index(drop=True)

# Apply SMOTE for oversampling the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, target)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Define the neural network model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
])

# Add hidden layers dynamically
num_hidden_layers = 5  # Increased number of hidden layers for complexity

for _ in range(num_hidden_layers):
    model.add(Dense(512, activation='relu'))  # Increased units in hidden layers
    model.add(Dropout(0.4))  # Adjusted dropout rate
    model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

# Compile the model with a lower learning rate
model.compile(optimizer=Nadam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=20, batch_size=2048, verbose=1, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr])

# Use model.predict to get predicted probabilities
y_pred_prob = model.predict(X_val)
# Threshold probabilities to get binary predictions
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute the accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Save the model
model.save('fraud_detection_model.h5')