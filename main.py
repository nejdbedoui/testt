import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
# Load data from CSV file
df = pd.read_csv('generated_data_knn0.65_lng0.64.csv')

# Convert gender column to string
df['gender'] = df['gender'].astype(str)

# Features and target
X = df.drop('category', axis=1)
y = df['category']

# Preprocessing
numeric_features = ['age', 'required_lvl']
categorical_features = ['gender', 'premium']

# Create column transformer with OneHotEncoder for categorical features and StandardScaler for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines for different models
pipelines = {
    'knn': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())]),
     'logistic_regression': Pipeline(
         steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    'knn': {
        'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    },
     'logistic_regression': {
         'classifier__C': [0.01, 0.1, 1, 10, 100],
         'classifier__solver': ['liblinear', 'lbfgs']
     }
}

# Plot distributions for numerical features
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['age'], kde=True, ax=axes[0])
axes[0].set_title('Age Distribution')
sns.histplot(df['required_lvl'], kde=True, ax=axes[1])
axes[1].set_title('Required Level Distribution')
plt.tight_layout()
plt.show()

# Plot distributions for categorical features
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x='gender', data=df, ax=axes[0])
axes[0].set_title('Gender Distribution')
sns.countplot(x='premium', data=df, ax=axes[1])
axes[1].set_title('Premium Distribution')
plt.tight_layout()
plt.show()


best_accuracy = 0
best_model = None

# Train each model 20 times and keep the best one
for iteration in range(1):
    print(f"Iteration {iteration + 1}")
    for model_name in pipelines:
        grid_search = GridSearchCV(pipelines[model_name], param_grids[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Best Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid_search.best_estimator_

# Print the best accuracy achieved
print(f"Best Accuracy Overall: {best_accuracy}")

# Function to predict book category
def predict_category(user_data, model):
    user_df = pd.DataFrame([user_data], columns=X.columns)
    predicted_category = model.predict(user_df)[0]
    return predicted_category

# Example user data
user_data = {
    'age': 23,
    'required_lvl': 12,
    'gender': 'Male',
    'premium': 1
}

# Get predicted category using the best model
predicted_category = predict_category(user_data, best_model)
print(f"Predicted Category: {predicted_category}")
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best model saved as 'best_model.pkl'")
sns.pairplot(df, hue='category')
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


