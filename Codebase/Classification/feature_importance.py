import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.svm import SVC
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Import Data

df = pd.read_csv("../../Features/all_features.csv")
# df_test = pd.read_csv("../../Citation Graph Data/Novelty_Paper/Features/all_features.csv")

# Create features and labels

X = df[df.columns[2:30]]
y = df["Label"].replace({1:0, 2:1, 3:1})


# Create pipeline for Oversampling using SMOTE alongwith Random Under Sampling

over_sample = SMOTE(sampling_strategy=0.5)
under_sample = RandomUnderSampler(sampling_strategy=0.6)

steps = [('o',over_sample),('u',under_sample)]
pipeline = Pipeline(steps=steps)

# Transform the dataset

X, y = pipeline.fit_resample(X, y)

# Convert to numpy array

X = X.values
y = y.values

# Random Forest

print("\nTraining Random Forest:")

classifier = RandomForestClassifier(criterion="entropy")

classifier.fit(X, y)

# Get Importance
importance = classifier.feature_importances_

features = df[df.columns[2:38]].columns

idx = sorted(range(len(importance)), key = lambda k:importance[k])
idx = list(reversed(idx))

features = [features[i] for i in idx]
importance = [importance[i] for i in idx]

df = pd.DataFrame(importance, features)
# df.columns = ['Features', 'Importance']
print(df)

f = open("feature_importance.csv", "w")

# print("Feature,Importance\n")
f.write("Feature,Importance\n")

for fe, i in zip(features, importance):
	# print("{} :\t\t\t {}".format(fe, i))
	f.write("{},{}\n".format(fe, i))

f.close()