import sys, os, re, io
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 
import csv
from math import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

citing = []
cited = []

cited_filepath="/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/XML/"
citing_filepath="/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/XML/"

data_filepath = '/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/CitationList.csv'

punctuation=[',','.','/','?','\'','\"','{','}','[',']','(',')','_','-','*','^','!']


analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
	score = analyser.polarity_scores(sentence)
	return score
	#print("{:-<40} {}".format(sentence, str(score)))

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


def vader_citance(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Citance")
		return "NA"

	citance = soup_citing.find_all('ref',target="#b"+str(BIB_ID))

	text=[]

	for cite in citance:
		text.append(re.sub("<.*?>", " ", str(cite.find_parent() )))

	vader_list = []

	for occur in text:
		vader_list.append( sentiment_analyzer_scores(occur) )

	return vader_list


################################################################################


with open(data_filepath, 'r') as f:
	reader = csv.reader(f)
	csv_list = list(reader)

cited = []
citing = []

pos_max = []
pos_avg = []

neg_max = []
neg_avg = []

neu_max = []
neu_avg = []

com_max = []
com_avg = []

print("Starting extraction and matching of features...\n")

time_start = time.time()

for ii in range(1, len(csv_list)):
	print(ii)

	# Citing & Cited Paper Path
	cited_path=citing_filepath+"/"+str(csv_list[ii][2])+".tei.xml"
	citing_path=cited_filepath+"/"+str(csv_list[ii][1])+".tei.xml"

	#print(csv_list[ii][2], csv_list[ii][1])

	# Cited Paper Name
	try:
		cited_paper_name = paper_name(cited_path)
		citing_paper_name = paper_name(citing_path)
		cited.append(str(csv_list[ii][1]))
		citing.append(str(csv_list[ii][2]))
	except FileNotFoundError:
		cited.append(str(csv_list[ii][1]))
		citing.append(str(csv_list[ii][2]))

		pos_max.append("FileNotFound")
		neg_max.append("FileNotFound")
		neu_max.append("FileNotFound")
		com_max.append("FileNotFound")
		pos_avg.append("FileNotFound")
		neg_avg.append("FileNotFound")
		neu_avg.append("FileNotFound")
		com_avg.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][1]))
		continue;


	# Create Soup Objects for Citing & Cited XML file


	citing_pointer = open(citing_path,'r',encoding='utf-8')
	soup_citing = BeautifulSoup(citing_pointer, 'html.parser')

	cited_pointer = open(cited_path,'r',encoding='utf-8')
	soup_cited = BeautifulSoup(cited_pointer, 'html.parser')

	# VADER Score of Citance (MAX & AVG)

	cwmd_common = vader_citance(soup_citing, soup_cited, csv_list[ii])
	
	if len(cwmd_common) == 0:
		pos_max.append("NA")
		neg_max.append("NA")
		neu_max.append("NA")
		com_max.append("NA")
		pos_avg.append("NA")
		neg_avg.append("NA")
		neu_avg.append("NA")
		com_avg.append("NA")
	elif cwmd_common == "NA":
		pos_max.append("FileNotFound")
		neg_max.append("FileNotFound")
		neu_max.append("FileNotFound")
		com_max.append("FileNotFound")
		pos_avg.append("FileNotFound")
		neg_avg.append("FileNotFound")
		neu_avg.append("FileNotFound")
		com_avg.append("FileNotFound")
	else:
		pos = []
		neg = []
		neu = []
		com = []
		for score in cwmd_common:
			pos.append(score['pos'])
			neg.append(score['neg'])
			com.append(score['compound'])
			neu.append(score['neu'])

		pos_max.append(max(pos))
		pos_avg.append(np.mean(pos))

		neg_max.append(max(neg))
		neg_avg.append(np.mean(neg))

		neu_max.append(max(neu))
		neu_avg.append(np.mean(neu))

		com_max.append(max(com))
		com_avg.append(np.mean(com))
		# citance_result_max.append(max(cwmd_common))
		# citance_result_avg.append(np.mean(cwmd_common))


time_end = time.time()

print("Time Elapsed in Extracting all features: {}\n".format(time_end-time_start))

print("Making CSV file...")

f = open("vader_features.csv","w")

f.write("{},{},{},{},{},{},{},{},{},{}\n".format("Citing", "Cited", "pos_max", "pos_avg", "neg_max", "neg_avg", "neu_max", "neu_avg", "com_max", "com_avg"))

for i in range(len(pos_max)):
	f.write("{},{},{},{},{},{},{},{},{},{}\n".format(citing[i], cited[i], pos_max[i], pos_avg[i], neg_max[i], neg_avg[i], neu_max[i], neu_avg[i], com_max[i], com_avg[i]))

f.close()

print("Done!\n")

print("Total time elapsed: {}".format(time.time() - time_start))