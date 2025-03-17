import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('train_transaction.csv')

# Separate features and target
features = df.drop('isFraud', axis=1)
target = df['isFraud']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Define the preprocessing function
def preprocess_data(df, preprocessor=None, fit=False):
    # Identify categorical and numerical columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

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
    if preprocessor is None:
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    if fit:
        return preprocessor.fit_transform(df), preprocessor
    else:
        return preprocessor.transform(df)

# Preprocess the training data
X_train_processed, preprocessor = preprocess_data(X_train, fit=True)

# Preprocess the testing data
X_test_processed = preprocess_data(X_test, preprocessor=preprocessor, fit=False)

# Load the trained neural network model
model = load_model('fraud_detection_model.h5')

# Make predictions using the neural network model
nn_predictions = model.predict(X_test_processed)

# Define the rule-based system
def rule_based_system(transaction, nn_prediction):
    # Rule 1: Transaction Amount Threshold
    if transaction['TransactionAmt'] > 10000 and nn_prediction > 0.5:
        return True
    
    # Rule 2: Frequency Threshold for Transactions
    card_transactions = X_test[X_test['card1'] == transaction['card1']]
    if len(card_transactions) > 100 and nn_prediction > 0.6:
        return True
    
    # Rule 3: Unusual Transaction Locations
    if transaction['card4'] == 'visa' and transaction['card6'] == 'debit' and transaction['addr2'] != 87 and nn_prediction > 0.5:
        return True
    
    # Rule 4: Suspicious Spending Behavior
    if transaction['C1'] > 10 and nn_prediction > 0.5:
        return True
    
    # If none of the rules are triggered, return False
    return False

# Apply the rule-based system and neural network to the test set
results = []

# Use tqdm to create a progress bar
for i in tqdm(range(len(X_test))):
    transaction = X_test.iloc[i]
    nn_prediction = nn_predictions[i]
    is_fraud = rule_based_system(transaction, nn_prediction)
    results.append(is_fraud)

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results, columns=['isFraud'])
#print(results_df)
# Calculate accuracy
correct_predictions = (results_df['isFraud'] == y_test.reset_index(drop=True)).sum()
total_samples = len(results_df)
accuracy = correct_predictions / total_samples

print("Correct Predictions:", correct_predictions)
print("Total Samples:", total_samples)
print("Accuracy:", accuracy)