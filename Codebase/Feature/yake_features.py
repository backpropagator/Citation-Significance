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

def count_common_keywords(w1, w2):
	count = 0
	for word in w1:
		if word in w2:
			count += 1

	return count

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


def extract_keywords(sentence):
	simple_kwextractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.8, windowsSize=2, top=50)
	keywords = simple_kwextractor.extract_keywords(sentence)
	keywords = [kw[1] for kw in keywords]
	return keywords



def kw_title(soup_citing, soup_cited, c_list):

	title_citing = soup_citing.title.string
	title_cited = soup_cited.title.string

	if title_cited == None or title_citing == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"


	kw_citing = extract_keywords(title_citing)
	kw_cited = extract_keywords(title_cited)

	#print(kw_cited, kw_citing)

	tkw_common = count_common_keywords(kw_cited, kw_citing)

	return tkw_common


def kw_abstract(soup_citing, soup_cited, c_list):

	if soup_citing.abstract.p == None or soup_cited.abstract.p == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	abstract_citing = soup_citing.abstract.p.string
	abstract_cited = soup_cited.abstract.p.string

	if len(abstract_citing) < 5 or len(abstract_cited) < 5:
		return 0

	kw_citing = extract_keywords(abstract_citing)
	kw_cited = extract_keywords(abstract_cited)

	akw_common = count_common_keywords(kw_cited, kw_citing)

	return akw_common

def kw_body(soup_citing, soup_cited, c_list):

	if soup_citing.body == None or soup_cited.body == None:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	body_citing = BeautifulSoup(str(soup_citing.body), "html.parser").get_text()
	body_cited = BeautifulSoup(str(soup_cited.body), "html.parser").get_text()

	body_citing = remove_punctuation(body_citing)
	body_cited = remove_punctuation(body_cited)

	kw_citing = extract_keywords(body_citing)
	kw_cited = extract_keywords(body_cited)

	#print(kw_citing)

	bkw_common = count_common_keywords(kw_cited, kw_citing)

	return bkw_common 

def correct(name):
	s = ""
	if name is not None:
		for c in name.get_text():
			if c not in punctuation:
				s += c

	return s

def get_ref_authors(soup):

	reference = soup.find('div',type="references")

	soup = BeautifulSoup(str(reference), 'html.parser')
	bibliography = soup.find_all("biblstruct")

	author_list = []

	for bib in bibliography:
		s = BeautifulSoup(str(bib), 'html.parser')

		authors = s.find_all("author")

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

	return author_list

def get_author_overlap(soup_citing, soup_cited, c_list):

	#print(c_list[2], c_list[1])

	auth_citing = get_ref_authors(soup_citing)
	auth_cited = get_ref_authors(soup_cited)

	auth_citing_set = set(auth_citing)
	auth_cited_set = set(auth_cited)

	if len(auth_citing_set) == 0 or len(auth_cited_set) == 0:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	overlap = len(auth_citing_set.intersection(auth_cited_set))
	ratio = overlap / len(auth_citing_set)

	return ratio


def get_ref_titles(soup):
	reference = soup.find('div',type="references")

	soup = BeautifulSoup(str(reference), 'html.parser')
	bibliography = soup.find_all("biblstruct")

	titles = []

	for bib in bibliography:
		s = BeautifulSoup(str(bib), 'html.parser')

		t = s.find("title")
		t = t.get_text()

		titles.append(t)

	return titles

def get_title_overlap(soup_citing, soup_cited, c_list):

	title_citing = get_ref_titles(soup_citing)
	title_cited = get_ref_titles(soup_cited)

	title_citing_set = set(title_citing)
	title_cited_set = set(title_cited)

	if len(title_citing_set) == 0 or len(title_cited_set) == 0:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	overlap = len(title_citing_set.intersection(title_cited_set))
	ratio = overlap / len(title_citing_set)

	return ratio


def get_ref_venues(soup):
	reference = soup.find('div',type="references")

	soup = BeautifulSoup(str(reference), 'html.parser')
	bibliography = soup.find_all("biblstruct")

	venues = []

	for bib in bibliography:
		s = BeautifulSoup(str(bib), 'html.parser')

		v = s.find_all("title")

		if len(v) > 1:
			venues.append(v[1].get_text())
		else:
			venues.append("Other")

	return venues

def get_venue_overlap(soup_citing, soup_cited, c_list):

	venue_citing = get_ref_venues(soup_citing)
	venue_cited = get_ref_venues(soup_cited)

	venue_citing_set = set(venue_citing)
	venue_cited_set = set(venue_cited)

	if len(venue_citing_set) == 0 or len(venue_cited_set) == 0:
		print(c_list[2], c_list[1], "File Not Parsed Correctly")
		return "NA"

	overlap = len(venue_citing_set.intersection(venue_cited_set))
	ratio = overlap / len(venue_citing_set)

	return ratio



#######################################################################################

with open(data_filepath, 'r') as f:
    reader = csv.reader(f)
    csv_list = list(reader)


title_result = []
abstract_result = []
body_result = []
auth_overlap_result = []
title_overlap_result = []
venue_overlap_result = []

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
		cited.append(str(csv_list[ii][1]))
	except FileNotFoundError:
		cited.append(str(csv_list[ii][1]))
		citing.append(str(csv_list[ii][2]))

		title_result.append("FileNotFound")
		abstract_result.append("FileNotFound")
		body_result.append("FileNotFound")
		auth_overlap_result.append("FileNotFound")
		title_overlap_result.append("FileNotFound")
		venue_overlap_result.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][1]))
		continue;

	# Citing Paper Name
	

	try:
		citing_paper_name = paper_name(citing_path)
		citing.append(str(csv_list[ii][2]))
	except FileNotFoundError:
		cited.append(str(csv_list[ii][1]))
		citing.append(str(csv_list[ii][2]))

		title_result.append("FileNotFound")
		abstract_result.append("FileNotFound")
		body_result.append("FileNotFound")
		auth_overlap_result.append("FileNotFound")
		title_overlap_result.append("FileNotFound")
		venue_overlap_result.append("FileNotFound")
		print("{},{}: {} file not found".format(csv_list[ii][2], csv_list[ii][1], csv_list[ii][2]))
		title_result.append("NA")
		continue;

	# Create Soup Objects for Citing & Cited XML file


	citing_pointer = open(citing_path,'r',encoding='utf-8')
	soup_citing = BeautifulSoup(citing_pointer, 'html.parser')

	cited_pointer = open(cited_path,'r',encoding='utf-8')
	soup_cited = BeautifulSoup(cited_pointer, 'html.parser')

	# Common Keywords between Titles


	tkw_common = kw_title(soup_citing, soup_cited, csv_list[ii])
	
	title_result.append(tkw_common)


	# Common Keywords between Abstracts


	akw_common = kw_abstract(soup_citing, soup_cited, csv_list[ii])

	abstract_result.append(akw_common)


	# Common Keywords between Bodies


	bkw_common = kw_body(soup_citing, soup_cited, csv_list[ii])

	body_result.append(bkw_common)

	# Bibliographic Author Overlap


	auth_overlap = get_author_overlap(soup_citing, soup_cited, csv_list[ii])
	
	auth_overlap_result.append(auth_overlap)

	# Bibliographic Title Overlap


	title_overlap = get_title_overlap(soup_citing, soup_cited, csv_list[ii])

	title_overlap_result.append(title_overlap)

	# Bibliographic Venue Overlap


	venue_overlap = get_venue_overlap(soup_citing, soup_cited, csv_list[ii])
	venue_overlap_result.append(venue_overlap)

time_end = time.time()

print("Time Elapsed in Extracting all features: {}\n".format(time_start-time_end))

print("Making CSV file...")

f = open("yake_features.csv","w")

f.write("{},{},{},{},{},{},{},{}\n".format("Citing", "Cited", "kw_title", "kw_abstract", "kw_body", "bib_Author_overlap", "bib_Title_overlap", "bib_Venue_overlap"))

for i in range(len(venue_overlap_result)):
	f.write("{},{},{},{},{},{},{},{}\n".format(citing[i], cited[i], title_result[i], abstract_result[i], body_result[i], auth_overlap_result[i], title_overlap_result[i], venue_overlap_result[i]))

f.close()

print("Done!\n")

print("Total time elapsed: {}".format(time.time() - time_start))
