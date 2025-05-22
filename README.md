# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data**: Import the dataset and separate features (X) and target (y).

2. **Split Data**: Divide into training (80%) and testing (20%) sets.

3. **Scale Features**: Standardize the features using `StandardScaler`.

4. **Define SVM Model**: Initialize a Support Vector Machine (SVM) classifier.

5. **Hyperparameter Grid**: Define a range of values for `C`, `kernel`, and `gamma` for tuning.

6. **Grid Search**: Perform Grid Search with Cross-Validation to find the best hyperparameters.

7. **Results Visualization**: Create a heatmap to show the mean accuracy for different combinations of hyperparameters.

8. **Best Model**: Extract the best model with optimal hyperparameters.

9. **Make Predictions**: Use the best model to predict on the test set.

10. **Evaluate Model**: Calculate accuracy and print the classification report.

## Program:

Developed by: YASHWANTH RAJA DURAI V
RegisterNumber: 212222040184
```
Program to implement SVM for food classification for diabetic patients.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('food_items_binary.csv')  # Replace with your dataset file

# Separate features and target
X = data.drop(columns=['class'])  # Replace 'class' with your target column name
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the SVM model
svm = SVC()

# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid_search.fit(X_train_scaled, y_train)

# Extract the results into a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Pivot the results for a heatmap
heatmap_data = results.pivot_table(index='param_kernel', columns='param_C', values='mean_test_score')

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
plt.title('Hyperparameter Tuning: Mean Test Accuracy')
plt.xlabel('C')
plt.ylabel('Kernel')
plt.show()

# Evaluate the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)
```

## Output:

![Screenshot 2025-05-15 182235](https://github.com/user-attachments/assets/5cc89e15-c6ee-44f5-a788-f44c9001c0a1)



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
