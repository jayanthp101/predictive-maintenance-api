import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

# File path (Ensure correct path format)
file_path = r"B:\predictive_main_bosch_data\predictive_maintenance.csv"

# Read dataset
try:
    df = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Dataset: {e}")
    exit()

# Encode categorical variables
le_type = LabelEncoder()
df['Type'] = le_type.fit_transform(df['Type'])

le_failure = LabelEncoder()
df['Failure Type'] = le_failure.fit_transform(df['Failure Type'])

# Define features and target
X = df[['Type', 'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = df['Failure Type']  # Multiclass classification

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best Model
model = grid_search.best_estimator_
print(f"✅ Best Parameters: {grid_search.best_params_}")

# Save Label Encoders & Scaler
joblib.dump(le_type, r"B:\predictive_main_bosch_data\type_label_encoder.pkl")
joblib.dump(le_failure, r"B:\predictive_main_bosch_data\failure_label_encoder.pkl")
joblib.dump(scaler, r"B:\predictive_main_bosch_data\feature_scaler.pkl")

# Save Trained Model
joblib.dump(model, r"B:\predictive_main_bosch_data\predictive_maintenance_model.pkl")
print("✅ AI Model Saved Successfully!")
