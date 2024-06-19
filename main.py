import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

# Load the data
data = pd.read_csv('test_set_features.csv')

# Separate features and target variables
X = data.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y = data[['xyz_vaccine', 'seasonal_vaccine']]

# Preprocess categorical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Define preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

# Create and evaluate the pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on test data
y_pred_proba = clf.predict_proba(X_test)

# Calculate the ROC AUC score
roc_auc_xyz = roc_auc_score(y_test['xyz_vaccine'], y_pred_proba[0][:, 1])
roc_auc_seasonal = roc_auc_score(y_test['seasonal_vaccine'], y_pred_proba[1][:, 1])
mean_roc_auc = np.mean([roc_auc_xyz, roc_auc_seasonal])

print(f'Mean ROC AUC: {mean_roc_auc}')

# Prepare the submission
test_data = pd.read_csv('test_data.csv')
X_test_final = test_data.drop(['respondent_id'], axis=1)
y_test_pred_proba = clf.predict_proba(X_test_final)

submission = pd.DataFrame({
    'respondent_id': test_data['respondent_id'],
    'xyz_vaccine': y_test_pred_proba[0][:, 1],
    'seasonal_vaccine': y_test_pred_proba[1][:, 1]
})

submission.to_csv('submission_format.csv', index=False)
