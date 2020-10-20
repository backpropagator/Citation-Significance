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
	# print(soup_cited.title)
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



def total_citation(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Abstract")
		return "NA"

	abstract_cited = soup_cited.abstract.p.string

	BIB_ID = find_bib_Id(soup_citing, soup_cited)

	if BIB_ID == "NA":
		print(c_list[2], c_list[1], "File Not Parsed Correctly, CItance")
		return "NA"

	citance = soup_citing.find_all('ref',target="#b"+str(BIB_ID))

	return len(citance)

def correct(name):
	s = ""
	if name is not None:
		for c in name.get_text():
			if c not in punctuation:
				s += c

	return s


def get_authors(soup):
	authors = soup.find('analytic').find_all('author')

	author_list = []

	for author in authors:
		name_soup = BeautifulSoup(str(author), 'html.parser')
		first = name_soup.find("forename", type = 'first')
		middle = name_soup.find("forename", type = 'middle')
		sur = name_soup.find("surname")

		name = ""

		name += correct(first)
		name += " "

		name += correct(middle)
		name += " "

		name += correct(sur)

		author_list.append(name)
		# print(a.persname)

	return author_list


def author_overlap(soup_citing, soup_cited, c_list):

	if soup_citing.analytic.author == None or soup_cited.analytic.author == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly, Author")
		return "NA"

	citing_author = get_authors(soup_citing)
	cited_author = get_authors(soup_cited)

	total_author = len(citing_author)

	auth_bool = 0
	auth_ratio = 0
	match = 0

	for i in citing_author:
		i=i.split(" ")
		i=set(i)
		for j in cited_author:
			j=j.split(" ")
			j=set(j)
			if(len(i.difference(j))==0):
				auth_bool = 1
				match = match + 1
	
	auth_ratio = match/total_author

	return auth_bool, auth_ratio

	




#######################################################################



with open(data_filepath, 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)

cited = []
citing = []

num_citation = []
auth_overlap = []
auth_ratio = []
cite_per_bib = []
cite_per_cite = []

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

		num_citation.append("FileNotFound")
		auth_overlap.append("FileNotFound")
		auth_ratio.append("FileNotFound")
		cite_per_bib.append("FileNotFound")
		cite_per_cite.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][1]))
		continue;


	# Create Soup Objects for Citing & Cited XML file


	citing_pointer = open(citing_path,'r',encoding='utf-8')
	soup_citing = BeautifulSoup(citing_pointer, 'html.parser')

	cited_pointer = open(cited_path,'r',encoding='utf-8')
	soup_cited = BeautifulSoup(cited_pointer, 'html.parser') 

	# Total number of citation

	num_cite = total_citation(soup_citing, soup_cited, csv_list[ii])
	num_citation.append(num_cite)

	# No. of citations/total no. of citations
	total_citations = len(soup_citing.find_all('ref'))
	cite_per_cite.append(num_cite/total_citations)

	# No. of citations/total no. of bib Items
	ref = soup_citing.find('div',type="references")
	ref = str(ref)
	ref = BeautifulSoup(ref,'html.parser')
	bib = ref.find_all('biblstruct')

	total_bib = len(bib)
	print(num_cite, total_bib)
	cite_per_bib.append(num_cite/total_bib)

	# Author Overlap
	auth_bool, ratio = author_overlap(soup_citing, soup_cited, csv_list[ii])
	auth_overlap.append(auth_bool)
	auth_ratio.append(ratio)


time_end = time.time()


print("Time Elapsed in Extracting all features: {}\n".format(time_end-time_start))

print("Making CSV file...")

f = open("basic_features.csv","w")

f.write("{},{},{},{},{},{},{}\n".format("Citing", "Cited","Total_Citation", "Author_Overlap", "Author_Overlap_Ratio", "Cite_per_Bib", "Cite_per_Citation"))

for i in range(len(citing)):
	f.write("{},{},{},{},{},{},{}\n".format(citing[i], cited[i], num_citation[i], auth_overlap[i], auth_ratio[i], cite_per_bib[i], cite_per_cite[i]))

f.close()

print("Done!\n")

print("Total time elapsed: {}".format(time.time() - time_start))




