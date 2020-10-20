import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import Data

df = pd.read_csv("../../Features/all_features.csv")


# Create features and labels

X = df[df.columns[2:30]]
y = df["Label"].replace({1:0, 2:1, 3:1})




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create pipeline for Oversampling using SMOTE alongwith Random Under Sampling

over_sample = SMOTE(sampling_strategy=0.31)
under_sample = RandomUnderSampler(sampling_strategy=0.31)

steps = [('o',over_sample),('u',under_sample)]
pipeline = Pipeline(steps=steps)

# Transform the dataset

X_train, y_train = pipeline.fit_resample(X_train, y_train) 

# Split in Training and Testing Data

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Convert to numpy array

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values



# SVM

print("SVM:")

classifier = SVC(kernel='rbf', gamma='scale')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


performance_dict = classification_report(y_test, y_pred, output_dict=True)['1']

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Precision: {}".format(performance_dict['precision']))
print("Recall: {}".format(performance_dict['recall']))
print("F1-Score: {}\n".format(performance_dict['f1-score']))
# print("Classification Reoport:\n {}\n".format( classification_report(y_test, y_pred, output_dict=True)['1'] ))
# print("Confusion Matrix:\n {}\n".format(confusion_matrix(y_test, y_pred)))


# Decision Tree

print("Decision Tree:")

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


performance_dict = classification_report(y_test, y_pred, output_dict=True)['1']

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Precision: {}".format(performance_dict['precision']))
print("Recall: {}".format(performance_dict['recall']))
print("F1-Score: {}\n".format(performance_dict['f1-score']))
# print("Classification Reoport:\n {}\n".format( classification_report(y_test, y_pred, output_dict=True)['1'] ))
# print("Confusion Matrix:\n {}\n".format(confusion_matrix(y_test, y_pred)))


# Random Forest

print("Random Forest:")

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

performance_dict = classification_report(y_test, y_pred, output_dict=True)['1']

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Precision: {}".format(performance_dict['precision']))
print("Recall: {}".format(performance_dict['recall']))
print("F1-Score: {}\n".format(performance_dict['f1-score']))
# print("Classification Reoport:\n {}\n".format( classification_report(y_test, y_pred, output_dict=True)['1'] ))
print("Confusion Matrix:\n {}\n".format(confusion_matrix(y_test, y_pred)))


# k-Nearest Neighbor

print("kNN:")

classifier = KNeighborsClassifier(n_neighbors=4)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


performance_dict = classification_report(y_test, y_pred, output_dict=True)['1']

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Precision: {}".format(performance_dict['precision']))
print("Recall: {}".format(performance_dict['recall']))
print("F1-Score: {}\n".format(performance_dict['f1-score']))
# print("Classification Reoport:\n {}\n".format( classification_report(y_test, y_pred, output_dict=True)['1'] ))
# print("Confusion Matrix:\n {}\n".format(confusion_matrix(y_test, y_pred)))