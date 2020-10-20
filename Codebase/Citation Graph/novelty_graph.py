import numpy as np
import pandas as pd
import csv
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

df_train = pd.read_csv("../../Features/all_features.csv")
df_test = pd.read_csv("../../Citation Graph Data/MENNDL/Features/all_features.csv")

# Create features and labels

X_train = df_train[df_train.columns[2:38]]
y_train = df_train["Label"].replace({1:0, 2:1, 3:1})

X_test = df_test[df_test.columns[2:38]]
# y_test = df_test["Label"].replace({1:0, 2:1, 3:1})

# Create pipeline for Oversampling using SMOTE alongwith Random Under Sampling

over_sample = SMOTE(sampling_strategy=0.5)
under_sample = RandomUnderSampler(sampling_strategy=0.6)

steps = [('o',over_sample),('u',under_sample)]
pipeline = Pipeline(steps=steps)

# Transform the dataset

X_train, y_train = pipeline.fit_resample(X_train, y_train)



# Convert to numpy array 

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
# y_test = y_test.values

with open("../../Citation Graph Data/MENNDL/Features/all_features.csv", 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)



# Random Forest

print("\nTraining Random Forest:")

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Done!\n")

# print(X_test, y_pred)

csv_list = csv_list[1:]
# print(len(csv_list),len(y_pred))

graph = {}


for i in range(len(csv_list)):
	# print(csv_list[i][0], csv_list[i][1], y_pred[i])

	if y_pred[i] == 1:
		if csv_list[i][0] not in graph.keys():
			graph[csv_list[i][0]] = []

		graph[csv_list[i][0]].append(csv_list[i][1])

print("Adjacency List of Citation Graph:")
for k in graph.keys():
	print("{} : {}".format(k, graph[k]))



