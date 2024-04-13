
from imblearn.over_sampling import ADASYN
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('nacc_AD.csv')


# Define your continuous and categorical variables
categorical_vars = ['SEX', 'Marital', 'familymember', 'Mom', 'Dad', 'AF', 'Stenting', 'CHF', 'AP', 'HVrepair',
                    'Hypertensive', 'Hypercholesterolemia', 'Diabetes', 'Hearing', 'DP', 'Neurosis', 'Sleepapnea',
                    'REM', 'Insomnia', 'Anticholinergic', 'Soporificdrug']
continuous_vars = ['EDUC', 'Smokingyears', 'sdp', 'dbp', 'Age', 'Drugstaken', 'BMI']

# Preprocessor for continuous features
continuous_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

# Preprocessor for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

important_categorical_vars = ['DP', 'REM', 'Neurosis', 'Mom', 'Stenting', 'SEX', 'Marital']
important_continuous_vars = ['EDUC', 'Age', 'BMI', 'sdp', 'Drugstaken']
# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, important_continuous_vars),
        ('cat', categorical_transformer, important_categorical_vars)
    ])

X = df[important_continuous_vars + important_categorical_vars]
y = df['AD']

# Fit the preprocessor to the entire dataset
preprocessor.fit(df)

# Apply ADASYN to the data only
adasyn = ADASYN(random_state=42, n_neighbors=2)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Split the balanced dataset into training and testing sets
X_train_7, X_test, y_train_7, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_7, y_train_7, test_size=0.2, random_state=42)

# Save datasets to disk
joblib.dump((X_train_7, y_train_7), 'train_7.joblib')
joblib.dump((X_test, y_test), 'test.joblib')
joblib.dump((X_train, y_train), 'train.joblib')
joblib.dump((X_val, y_val), 'val.joblib')

# Plot the distribution of the outcome variable in the original data set
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y)
plt.title('Original Data Set')
plt.ylabel('Count')

# Print the counts for each class in the original data set
print("Counts in original data set:")
print(Counter(y))

# Plot the distribution of the outcome variable in the balanced training set
plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled)
plt.title('Balanced Data Set')
plt.ylabel('Count')

# Print the counts for each class in the balanced training set
print("Counts in balanced training set:")
print(Counter(y_train_7))

plt.tight_layout()
plt.show()

# Print the counts for the Balanced set
print('Balanced dataset shape:', Counter(y_resampled))

# Print the counts for the validation set
print('Validation dataset shape:', Counter(y_val))

# Print the counts for the resampled training set
print('Resampled training dataset shape:', Counter(y_train))
