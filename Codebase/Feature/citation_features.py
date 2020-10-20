import sys, os, re, io
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 
import csv
from math import *
import yake
import time

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


def get_citance_length(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	citance = soup_citing.find_all('ref',target="#b"+str(BIB_ID))

	text=[]

	for cite in citance:
		text.append(re.sub("<.*?>", " ", str(cite.find_parent() )))

	length = []

	for occur in text:
		s = occur.split(" ")
		length.append(len(s))

	return np.average(length)


def get_diff_occurence(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	sections = soup_citing.find_all('head', n=True)

	occurence = []

	for i in range(len(sections)-1):
		s = str(soup_citing)

		start = str(sections[i])
		end = str(sections[i+1])

		section = s[s.find(start)+len(start):s.rfind(end)]

		ind='"#b'+str(BIB_ID)+'"'

		occurence.append(section.count(ind))

	return len(occurence)


def get_occurence_in_groups(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	paras = soup_citing.find_all('p')

	groups = []

	for para in paras:
		refs = para.find_all('ref')

		tmp = []

		for ref in refs:
			if(ref.has_attr('target') and ref.has_attr('type')):
				if(ref['type']=='bibr' and ref.has_attr('target')):
					if ref.has_attr('target'):
						tmp.append(str(ref['target']))
					else:
						tmp.append(str(1000))
			else:
				tmp.append('empty')

		groups.append(list(set(tmp)))

	occur = 0
	for refs in groups:
		if(len(refs)>1 and 'empty' not in refs):
			s = "#b"+str(BIB_ID)
			if s in refs and len(refs) > 1:
				occur = 1
				break

	return occur


def get_references(soup):
	ref = soup.find('div',type="references")
	ref = str(ref)
	ref = BeautifulSoup(ref,'html.parser')
	bib = ref.find_all('biblstruct')

	titles = []

	for b in bib:
		titles.append(b.title.string)

	return titles






def get_reference_overlap(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	ref_citing = get_references(soup_citing)
	ref_cited = get_references(soup_cited)

	count = 0

	for paper in ref_citing:
		if paper in ref_cited:
			count += 1

	ratio = count/len(ref_citing)

	return ratio



def get_distance_between_citance(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	citance = soup_citing.find_all('ref',target="#b"+str(BIB_ID))

	if len(citance) == 1:
		return 0

	length = []

	for i in range(len(citance)-1):
		s = str(soup_citing)

		start = str(citance[i])
		end = str(citance[i+1])

		text = s[s.find(start)+len(start):s.rfind(end)]

		length.append(len(text.split(" ")))

	return np.mean(length)







#######################################################################

with open(data_filepath, 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)

cited = []
citing = []

citance_length = []
diff_sections = []
occur_in_groups = []
word_distance = []
ref_overlap_ratio = []


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

		citance_length.append("FileNotFound")
		diff_sections.append("FileNotFound")
		occur_in_groups.append("FileNotFound")
		temporal_distance.append("FileNotFound")
		ref_overlap_ratio.append("FileNotFound")
		word_distance.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][1]))
		continue;


	# Create Soup Objects for Citing & Cited XML file


	citing_pointer = open(citing_path,'r',encoding='utf-8')
	soup_citing = BeautifulSoup(citing_pointer, 'html.parser')

	cited_pointer = open(cited_path,'r',encoding='utf-8')
	soup_cited = BeautifulSoup(cited_pointer, 'html.parser')

	# Length of Citance
	l = get_citance_length(soup_citing, soup_cited, csv_list[ii])
	citance_length.append(l)

	# Occurence in different Sections
	different_section = get_diff_occurence(soup_citing, soup_cited, csv_list[ii])
	diff_sections.append(different_section)

	# Occurence in Group
	occur = get_occurence_in_groups(soup_citing, soup_cited, csv_list[ii])
	occur_in_groups.append(occur)

	# Common References / Reference Overlap
	ratio = get_reference_overlap(soup_citing, soup_cited, csv_list[ii])
	ref_overlap_ratio.append(ratio)

	# Distance b/w Citance (no. of words)
	distance = get_distance_between_citance(soup_citing, soup_cited, csv_list[ii])
	word_distance.append(distance)



time_end = time.time()


print("Time Elapsed in Extracting all features: {}\n".format(time_end-time_start))

print("Making CSV file...")

f = open("citance_features.csv","w")

f.write("{},{},{},{},{},{},{}\n".format("Citing", "Cited","Citation_Length", "Occurence_In_Sections", "Occurs_In_Group", "Reference_Overlap", "Distance_Between_Citance"))

for i in range(len(citing)):
	f.write("{},{},{},{},{},{},{}\n".format(citing[i], cited[i], citance_length[i], diff_sections[i], occur_in_groups[i], ref_overlap_ratio[i], word_distance[i]))

f.close()

print("Done!\n")

print("Total time elapsed: {}".format(time.time() - time_start))





	