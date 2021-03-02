# Citation-Significance
This repository consists of Codes used for QSS Submission

The directories are organized as follows:

- Codebase

	- Citation Graph
		- novelty_graph.py : code to create the SCG (Significant Citation Graph)

	- Classification
		- classify.py : code to get the classification results using the extracted features
		- feature_importance.py : code to get the information gain of each feature

	- Feature
		- Word_Mover_Distance.py : Module to calculate the Word Mover's Distance
		- basic_features.py : Code to extract features F1 - F5
		- citation_features.py : Code to extract features F6 - F7 and F35 - F36
		- tfidf_features.py : Code to extract tf-idf features (F8 - F13)
		- vader_features.py : Code to extract VADER Sentiment Index features (F27 - F34)
		- wmd_features.py : Code to extract Word Mover's Distance features (F22 - F26)
		- yake_features.py : Code to extract YAKE features (F19 - F21)

**Notes**
1. All the codes were run using Python 3.7
2. Update the location of directory in the codes according to your directory setup.
3.  Location of GROBID parsed XML files should be put in codes in Feature Directory
4. Location of extracted features should be put in codes in Citation Graph and Classification directory.
