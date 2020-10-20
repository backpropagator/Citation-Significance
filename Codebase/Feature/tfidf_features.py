import sys, os, re, io
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 
import csv
from math import *
from gensim.models import Word2Vec, KeyedVectors
from Word_Mover_Distance import *
import time
from sklearn.feature_extraction.text import TfidfVectorizer


citing = []
cited = []

cited_filepath="/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/XML/"
citing_filepath="/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/XML/"

data_filepath = '/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/CitationList.csv'

punctuation=[',','.','/','?','\'','\"','{','}','[',']','(',')','_','-','*','^','!']


def remove_punctuation(sentence):
	for c in sentence:
		if c in punctuation:
			sentence = sentence.replace(c,"")

	return sentence

def paper_name(path):
	html_report_part1 = open(path,'r',encoding='utf-8')
	soup = BeautifulSoup(html_report_part1, 'html.parser')
	ref=soup.find('titlestmt')
	ref=str(ref)
	ref=BeautifulSoup(ref,'html.parser')
	ref=ref.find('title',type='main',level='a')
	if(ref!=None):
		if(len(ref.contents)>0):
			r=ref.contents[0]
			return r
		else:
			return "NA"
	else:
		return "NA"

def find_bib_Id(soup_citing, soup_cited):

	title_cited = soup_cited.title.string.lower()

	ref = soup_citing.find('div',type="references")
	ref = str(ref)
	ref = BeautifulSoup(ref,'html.parser')
	bib = ref.find_all('biblstruct')
	bib_items = []

	for i in range(len(bib)):
		ref = BeautifulSoup(str(bib[i]), "html.parser")
		ref=ref.find('title',type="main")
		
		if ref != None:
			if ref.contents[0] != " ":
				bib_items.append(ref.contents[0].lower())
			else:
				bib_items.append("NA")
		else:
			bib_items.append("NA")

	for i in range(len(bib_items)):
		set1 = set(bib_items[i].split(" "))
		set2 = set(title_cited.split(" "))

		if len(set2.difference(set1)) <= 1 and bib_items[i] != "NA":
			return i

	return "NA"


def tf_idf_Similarity(doc1, doc2):
	corpus = [doc1, doc2]

	vect = TfidfVectorizer(min_df=1, stop_words="english")
	tfidf = vect.fit_transform(corpus)

	pairwise_similarity = tfidf * tfidf.T

	pairwise_similarity = pairwise_similarity.toarray()

	return pairwise_similarity[0][1]


def tfidf_title(soup_citing, soup_cited, c_list):

	title_citing = soup_citing.title.string
	title_cited = soup_cited.title.string

	if title_cited == None or title_citing == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	return tf_idf_Similarity(title_citing, title_cited)

def tfidf_abstract(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	abstract_citing = soup_citing.abstract.p.string
	abstract_cited = soup_cited.abstract.p.string

	return tf_idf_Similarity(abstract_citing, abstract_cited)


def tfidf_citance_abstract(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	abstract_cited = soup_cited.abstract.p.string

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	citance = soup_citing.find_all('ref',target="#b"+str(BIB_ID))

	text=[]

	for cite in citance:
		text.append(re.sub("<.*?>", " ", str(cite.find_parent() )))

	wmd_list = []

	for occur in text:
		wmd_list.append( tf_idf_Similarity(abstract_cited, occur) )

	return wmd_list


def tfidf_citance_title(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	abstract_cited = soup_cited.title.string

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	citance = soup_citing.find_all('ref',target="#b"+str(BIB_ID))

	text=[]

	for cite in citance:
		text.append(re.sub("<.*?>", " ", str(cite.find_parent() )))

	wmd_list = []

	for occur in text:
		wmd_list.append( tf_idf_Similarity(abstract_cited, occur) )

	return wmd_list



#############################################################################


with open(data_filepath, 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)

cited = []
citing = []

title_similarity = []
abstract_similarity = []
citance_title_max = []
citance_title_avg = []
citance_abstract_max = []
citance_abstract_avg = []

print("Starting extraction and matching of features...\n")

time_start = time.time()

for ii in range(1, len(csv_list)):
	print(ii)

	# Citing & Cited Paper Path
	cited_path=citing_filepath+"/"+str(csv_list[ii][2])+".tei.xml"
	citing_path=cited_filepath+"/"+str(csv_list[ii][1])+".tei.xml"


	# Cited Paper Name
	try:
		cited_paper_name = paper_name(cited_path)
		citing_paper_name = paper_name(citing_path)
		cited.append(str(csv_list[ii][1]))
		citing.append(str(csv_list[ii][2]))
	except FileNotFoundError:
		cited.append(str(csv_list[ii][1]))
		citing.append(str(csv_list[ii][2]))

		title_similarity.append("FileNotFound")
		abstract_similarity.append("FileNotFound")
		citance_title_max.append("FileNotFound")
		citance_title_avg.append("FileNotFound")
		citance_abstract_max.append("FileNotFound")
		citance_abstract_avg.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][1]))
		continue;

	# Create Soup Objects for Citing & Cited XML file


	citing_pointer = open(citing_path,'r',encoding='utf-8')
	soup_citing = BeautifulSoup(citing_pointer, 'html.parser')

	cited_pointer = open(cited_path,'r',encoding='utf-8')
	soup_cited = BeautifulSoup(cited_pointer, 'html.parser') 


	# Similarity between Titles
	title_sim = tfidf_title(soup_citing, soup_cited, csv_list[ii])
	title_similarity.append(title_sim)

	# Similarity between Abstracts
	abstract_sim = tfidf_abstract(soup_citing, soup_cited, csv_list[ii])
	abstract_similarity.append(abstract_sim)

	# Similarity between Citance & Title (Max & Avg)
	citance_title_sim = tfidf_citance_title(soup_citing, soup_cited, csv_list[ii])

	citance_title_avg.append(np.max(citance_title_sim))
	citance_title_max.append(np.mean(citance_title_sim))

	# Similarity between Citance & Abstracts (Max & Avg)
	citance_abstract_sim = tfidf_citance_abstract(soup_citing, soup_cited, csv_list[ii])

	citance_abstract_avg.append(np.max(citance_abstract_sim))
	citance_abstract_max.append(np.mean(citance_abstract_sim))



time_end = time.time()


print("Time Elapsed in Extracting all features: {}\n".format(time_end-time_start))

print("Making CSV file...")

f = open("tfidf_features.csv","w")

f.write("{},{},{},{},{},{},{},{}\n".format("Citing", "Cited","tfidf_Title", "tfidf_Abstract", "tfidf_Citance_Title (max)", "tfidf_Citance_Title (avg)", "tfidf_Citance_Abstract (max)", "tfidf_Citance_Abstract (avg)"))

for i in range(len(citing)):
	f.write("{},{},{},{},{},{},{},{}\n".format(citing[i], cited[i], title_similarity[i], abstract_similarity[i], citance_title_max[i], citance_title_avg[i], citance_abstract_max[i], citance_abstract_avg[i]))

f.close()

print("Done!\n")

print("Total time elapsed: {}".format(time.time() - time_start))