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

# Get data (only train data)
path = os.path.join('PreparedData', 'train.csv')
df = pd.read_csv(path)


# Feature Engineering (one hot encode)
df = pd.get_dummies(df, drop_first=True)


# Test train split
X_train, X_test, y_train, y_test = train_test_split(df.drop('moved_after_2019', axis=1), df['moved_after_2019'], test_size=0.1, random_state=42)


rf_model = RandomForestClassifier(n_estimators=550, max_depth=12, random_state=40)
rf_model.fit(X_train, y_train)


rf_model.score(X_test, y_test)


# Load df_train
path = os.path.join('PreparedData', 'train.csv')
df_train = pd.read_csv(path)
df_train = pd.get_dummies(df_train, drop_first=True)

path = os.path.join('PreparedData', 'test.csv')
df_test = pd.read_csv(path)
df_test = pd.get_dummies(df_test, drop_first=True)



X = df_train.drop('moved_after_2019', axis=1).values
y = df_train['moved_after_2019'].values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the cross-validation object
kfold = KFold(n_splits=10)

# Results dictionary
results = {}


# Initialize the Random Forest Classifier
n_estimators_list = [200, 300, 500, 600, 800, 1000, 1250]
max_depth_list = [3, 5, 7, 10, 20, 25, 30, 42]

for n_estimators in n_estimators_list:
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
with open("results.txt", "w") as f:
    for key, value in results.items():
        f.write("n_estimators: %s, max_depth: %s, mean validation accuracy: %s, mean training accuracy: %s, mean loss: %s", 
                key[0], key[1], value[0], value[1], value[2])


# Top 3 results based on mean validation accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
# Print with colors, the title
print("\033[1m" + "* Top 3 results based on mean validation accuracy *" + "\033[0m")
for i in range(3):
    print("n_estimators:", sorted_results[i][0][0])
    print("max_depth:", sorted_results[i][0][1])
    print("Mean validation accuracy:", sorted_results[i][1][0])
    print("Mean training accuracy:", sorted_results[i][1][1])
    print("Mean loss:", sorted_results[i][1][2])
    print("=============================================")









