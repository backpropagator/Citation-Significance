import sys, os, re, io
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 
import csv
from math import *
from gensim.models import Word2Vec, KeyedVectors
from Word_Mover_Distance import *
import time


citing = []
cited = []

cited_filepath="/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/XML/"
citing_filepath="/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/XML/"

data_filepath = '/home/phd/tirthankar2/Piyush/Citation_Graph_Data/MENNDL/CitationList.csv'

punctuation=[',','.','/','?','\'','\"','{','}','[',']','(',')','_','-','*','^','!']

if not os.path.exists('/home/phd/tirthankar2/Piyush/w2v/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
    
model = KeyedVectors.load_word2vec_format('/home/phd/tirthankar2/Piyush/w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)
model.init_sims(replace=True)

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


def get_wmd(text1, text2):
	return word_movers_distance(model, text1, text2)

def wmd_title(soup_citing, soup_cited, c_list):

	title_citing = soup_citing.title.string
	title_cited = soup_cited.title.string

	if title_cited == None or title_citing == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	return get_wmd(title_citing, title_cited)

def wmd_abstract(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	abstract_citing = soup_citing.abstract.p.string
	abstract_cited = soup_cited.abstract.p.string

	return get_wmd(abstract_citing, abstract_cited)


def wmd_body(soup_citing, soup_cited, c_list):

	if soup_citing.body == None or soup_cited.body == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	body_citing = BeautifulSoup(str(soup_citing.body), "html.parser").get_text()
	body_cited = BeautifulSoup(str(soup_cited.body), "html.parser").get_text()

	body_citing = remove_punctuation(body_citing)
	body_cited = remove_punctuation(body_cited)

	return get_wmd(body_citing, body_cited)

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


def wmd_citance(soup_citing, soup_cited, c_list):

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
		wmd_list.append( get_wmd(abstract_cited, occur) )

	return wmd_list

#######################################################################

with open(data_filepath, 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)

cited = []
citing = []

title_result = []
abstract_result = []
body_result = []
citance_result_max = []
citance_result_avg = []

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

		title_result.append("FileNotFound")
		abstract_result.append("FileNotFound")
		body_result.append("FileNotFound")
		citance_result_max.append("FileNotFound")
		citance_result_avg.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][1]))
		continue;

	# Citing Paper Name
	

	# try:
	# 	citing_paper_name = paper_name(citing_path)
	# 	citing.append(str(csv_list[ii][2]))
	# except FileNotFoundError:
	# 	cited.append(str(csv_list[ii][1]))
	# 	citing.append(str(csv_list[ii][2]))

	# 	title_result.append("FileNotFound")
	# 	abstract_result.append("FileNotFound")
	# 	body_result.append("FileNotFound")
	# 	citance_result_max.append("FileNotFound")
	# 	citance_result_avg.append("FileNotFound")
	# 	print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][2]))
	# 	title_result.append("NA")
	# 	continue;

	# Create Soup Objects for Citing & Cited XML file


	citing_pointer = open(citing_path,'r',encoding='utf-8')
	soup_citing = BeautifulSoup(citing_pointer, 'html.parser')

	cited_pointer = open(cited_path,'r',encoding='utf-8')
	soup_cited = BeautifulSoup(cited_pointer, 'html.parser') 


	# WMD between Titles


	twmd_common = wmd_title(soup_citing, soup_cited, csv_list[ii])
	
	title_result.append(twmd_common)


	# WMD between Abstracts


	awmd_common = wmd_abstract(soup_citing, soup_cited, csv_list[ii])

	abstract_result.append(awmd_common)


	# WMD between Bodies


	bwmd_common = wmd_body(soup_citing, soup_cited, csv_list[ii])

	body_result.append(bwmd_common)

	# WMD between Citance & Abstract (Max & Average)

	cwmd_common = wmd_citance(soup_citing, soup_cited, csv_list[ii])
	if len(cwmd_common) == 0:
		cwmd_common = [0]
	#print(cwmd_common)

	if cwmd_common == "NA":
		citance_result_max.append("NA")
		citance_result_avg.append("NA")
	else:
		citance_result_max.append(max(cwmd_common))
		citance_result_avg.append(np.mean(cwmd_common))

time_end = time.time()

print("Time Elapsed in Extracting all features: {}\n".format(time_end-time_start))

print("Making CSV file...")

f = open("wmd_features.csv","w")

f.write("{},{},{},{},{},{},{}\n".format("Citing", "Cited", "wmd_title", "wmd_abstract", "wmd_body", "wmd_citance_max", "wmd_citance_avg"))

for i in range(len(title_result)):
	f.write("{},{},{},{},{},{},{}\n".format(citing[i], cited[i], title_result[i], abstract_result[i], body_result[i], citance_result_max[i], citance_result_avg[i]))

f.close()

print("Done!\n")

print("Total time elapsed: {}".format(time.time() - time_start))