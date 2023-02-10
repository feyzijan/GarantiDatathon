# This script is used to train a random forest model and do cross validation
# import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# progress bar
from tqdm import tqdm

# Go two levels up
os.chdir('../..')

# Load df_train
path = os.path.join('PreparedData', 'train.csv')
df_train = pd.read_csv(path)
df_train = pd.get_dummies(df_train, drop_first=True)

# Make user_id the index
df_train = df_train.set_index('user_id')

# Show collumns
print(df_train.columns)

# Drop the target variable from the training set
X = df_train.drop('moved_after_2019', axis=1).values
y = df_train['moved_after_2019'].values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the cross-validation object
kfold = KFold(n_splits=10)

# Results dictionary
results = {}


# Initialize the Random Forest Classifier
n_estimators_list = [100, 250, 500, 1000, 2000, 3000, 5000]
max_depth_list = [3, 5, 7, 10, 14, 18, 20, 22, 24, 26, 28, 30, 42, 50, 85]

for n_estimators in tqdm(n_estimators_list):
    for max_depth in max_depth_list:
        # Show information
        print("Performing cross validation with the following parameters:")
        print("n_estimators:", n_estimators)
        print("max_depth:", max_depth)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        # Train and evaluate the model using cross-validation
        losses = []
        validation_accuracies = []
        training_accuracies = []

        # Loop over the folds
        for train_index, val_index in kfold.split(X_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
            
            clf.fit(X_train_cv, y_train_cv)
            y_pred = clf.predict(X_val_cv)
            y_pred_proba = clf.predict_proba(X_val_cv)
            accuracy = accuracy_score(y_val_cv, y_pred)
            validation_accuracies.append(accuracy)

            los = log_loss(y_val_cv, y_pred_proba)
            losses.append(los)

            training_accuracy = accuracy_score(y_train_cv, clf.predict(X_train_cv))
            training_accuracies.append(training_accuracy)
        # Average the accuracy across all folds
        mean_accuracy = sum(validation_accuracies) / len(validation_accuracies)
        print("Mean validation accuracy across all folds:", mean_accuracy)
        # Average the training accuracy across all folds
        mean_training_accuracy = sum(training_accuracies) / len(training_accuracies)
        print("Mean training accuracy across all folds:", mean_training_accuracy)

        # Average losses across all folds
        mean_loss = sum(losses) / len(losses)
        print("Mean loss across all folds:", mean_loss)
        # Store the results
        results[(n_estimators, max_depth)] = (mean_accuracy, mean_training_accuracy, mean_loss)
        print("=============================================")

# Save ALL the results to a file, nicely formatted
with open("results2.txt", "w") as f:
    for key, value in results.items():
        f.write("n_estimators: " + str(key[0]) + "\n")
        f.write("max_depth: " + str(key[1]) + "\n")
        f.write("Mean validation accuracy: " + str(value[0]) + "\n")
        f.write("Mean training accuracy: " + str(value[1]) + "\n")
        f.write("Mean loss: " + str(value[2]) + "\n")
        f.write("=============================================" + "\n")


# Top 3 results based on mean validation accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
# Print with green color, the title
print("\033[92m" + "Top 3 results based on mean validation accuracy" + "\033[0m")
breaker = 0
for i in range(len(sorted_results)):
    if breaker == 3:
        break
    print("n_estimators:", sorted_results[i][0][0])
    print("max_depth:", sorted_results[i][0][1])
    print("Mean validation accuracy:", sorted_results[i][1][0])
    print("Mean training accuracy:", sorted_results[i][1][1])
    print("Mean loss:", sorted_results[i][1][2])
    print("=============================================")
    breaker += 1









