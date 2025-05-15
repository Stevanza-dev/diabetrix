import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Using joblib instead of pickle for scikit-learn objects

# --- Configuration ---
DATASET_PATH = 'Diabetes.xlsx'
MODEL_FILENAME = 'model/diabetes_ensemble_model.joblib'
SCALER_FILENAME = 'model/scaler.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Load Data ---
print("Loading dataset...")
try:
    df = pd.read_excel(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: The dataset file '{DATASET_PATH}' was not found.")
    print("Please download and save it as 'Diabetes.xlsx' in the same directory as this script.")
    exit()

print("Dataset loaded successfully.")
print("Dataset head:\n", df.head())
print("\nDataset info:")
df.info()
print("\nDataset description:\n", df.describe())

# --- Exclude 'Id' column if exists ---
if 'Id' in df.columns:
    print("Excluding 'Id' column from the dataset...")
    df = df.drop('Id', axis=1)
    print("Updated columns:", df.columns.tolist())

# --- Preprocessing ---
print("\nPreprocessing data...")
# Define columns where 0 should be treated as missing
cols_to_replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN in specified columns
for col in cols_to_replace_zero:
    df[col] = df[col].replace(0, np.nan)

# Impute NaN values with the mean of each column
for col in cols_to_replace_zero:
    df[col] = df[col].fillna(df[col].mean())

print("\nMissing values after imputation (should be 0 for replaced columns):")
print(df[cols_to_replace_zero].isnull().sum())

# Define features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Initialization ---
print("\nInitializing classifiers...")
log_reg = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, penalty='l1') # Added some hyperparameters
rf_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10, min_samples_leaf=5) # Added some hyperparameters
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1) # Added some hyperparameters

# --- Ensemble Model (Soft Voting) ---
print("Creating Voting Classifier (soft voting)...")
# Ensure all classifiers can produce probability estimates
if not hasattr(log_reg, 'predict_proba') or \
   not hasattr(rf_clf, 'predict_proba') or \
   not hasattr(xgb_clf, 'predict_proba'):
    raise TypeError("One or more classifiers do not support predict_proba, which is required for soft voting.")

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('rf', rf_clf),
        ('xgb', xgb_clf)
    ],
    voting='soft' # Use soft voting
)

# --- Model Training ---
print("Training the ensemble model...")
voting_clf.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- Model Evaluation ---
print("\nEvaluating the model on the test set...")
y_pred_ensemble = voting_clf.predict(X_test_scaled)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {accuracy_ensemble:.4f}")
print("\nEnsemble Model Classification Report:\n", classification_report(y_test, y_pred_ensemble))

# Evaluate individual classifiers for comparison (optional)
print("\nEvaluating individual classifiers on the test set for comparison:")
for clf_name, clf_instance in [('Logistic Regression', log_reg), ('Random Forest', rf_clf), ('XGBoost', xgb_clf)]:
    # Re-fit individual classifiers if they were not part of a pipeline that got fitted
    # For standalone instances like here, they need to be fitted individually before prediction.
    # However, the VotingClassifier fits clones of these internally.
    # For a fair comparison, fit them on the same training data if they haven't been.
    # Here, we'll fit them again for clarity, though for VotingClassifier, internal clones are used.
    clf_instance.fit(X_train_scaled, y_train)
    y_pred_individual = clf_instance.predict(X_test_scaled)
    accuracy_individual = accuracy_score(y_test, y_pred_individual)
    print(f"{clf_name} Accuracy: {accuracy_individual:.4f}")
    # print(f"{clf_name} Classification Report:\n", classification_report(y_test, y_pred_individual))


# --- Save Model and Scaler ---
print(f"\nSaving the trained model to {MODEL_FILENAME}...")
joblib.dump(voting_clf, MODEL_FILENAME)
print("Model saved.")

print(f"Saving the scaler to {SCALER_FILENAME}...")
joblib.dump(scaler, SCALER_FILENAME)
print("Scaler saved.")

print("\nTraining and saving process finished.")
