import json
import numpy as np
import os
import pandas as pd
import string
import pickle


label_to_index = {"Author": 1,
"Author_etal": 2,
"Title": 3,
"Source": 4,
"Vol": 6,
"Issue": 7,
"Page_b": 8,
"Page_e": 9,
"Number": 10,
"Identifier": 11}

year_label = "Year"
title_label = "Title"
page_label = "Page"
author_label = "Author"
volume_label = "Vol"
issue_label = "Issue"
source_label = "Source"
identifier_label = "Identifier"
number_label = "Number"

ref_with_no_year = []
WOS = {}
arr = ["–", "--", "- -", "-", "_", ":", "to", ",", "To", "and", "through", "And", "Through"]
identifier_tokens = ["doi :", "DOI :","doi=", "doi "]


def clean_punc_from_str_list(list0):
    list1 = []
    for s in list0:
        s = s.lower()
        s = s.strip()
        s = s.replace(" & ", " and ")

        for c in string.punctuation:
            s = s.replace(c," ")
            s = s.replace("  ", " ")

        s = s.replace("   ", " ")
        s = s.replace("  ", " ")
        s = s.strip()
        list1.append(s)
    list1 = np.array(list1).reshape(list0.shape)
    return list1

def modify_page(b_page, e_page):

    n=len(e_page)
    t=len(b_page)
    page=b_page[0:t-n]+e_page

    return page

def get_name_format_excel(name_str):

    # remove - from names??
    name_str = name_str.replace(",", "")
    name_str = name_str.replace(" - ", "")
    name_str = name_str.replace("- ", "")
    name_str = name_str.replace(" -", "")
    name_str = name_str.replace("-", "")
    name_str = name_str.replace("  ", " ")
    name_str = name_str.replace(" .", ".")

    name_tokens = name_str.split(" ")
    initials = ""
    main_name = ""

    for token in name_tokens:
        if "." in token or len(token)==1 or (len(token)<=2 and token.isupper()):
            initials += token + " "

        else:
            main_name += token + " "

    initials = initials.replace(".", "")
    initials = initials.replace(" ","")
    name = ""
    if main_name!="":
        if initials!="":
            name = str.lower(main_name.strip() + ", " + initials.strip()).strip()
        else:
            name = str.lower(main_name.strip())

    elif initials!="":
        name = str.lower(initials.strip())


    return name


def get_cleaned_pages(value):

    value = value.strip()
    found = False

    for punc in arr:
        if punc in value:
            value = value.replace(" ","")
            tmp = value.split(punc)
            found = True
            break

    if found == False:
        value = value.replace("    "," ")
        value = value.replace("   "," ")
        value = value.replace("  "," ")
        tmp = value.split(" ")

    if len(tmp) == 1:
        b_page = tmp[0]
        return b_page, "NONE"

    elif len(tmp)==2:
        b_page = tmp[0]
        e_page = tmp[1]

        if len(e_page) < len(b_page):
            e_page = modify_page(b_page, e_page)
        return b_page, e_page

    print("**********")
    print("Error in page")
    print(tmp)
    print(value)

    return "NONE", "NONE"

def check_abbrev(str_1, str_2):
    if str_1=="na" or str_1=="" or str_2=="na" or str_2=="":
        return False
    str_1_tokens = str_1.split(" ")
    str_2_tokens = str_2.split(" ")

    if len(str_1_tokens) != len(str_2_tokens):
        return False

    for token_1, token_2 in zip(str_1_tokens, str_2_tokens):
        res_1 = token_1.find(token_2)
        res_2 = token_2.find(token_1)

        if not (res_2==0 or res_1==0):
            return False

    return True

def clean_punc_from_str(s):
    s = s.lower()
    s = s.strip()
    s = s.replace(" & ", " and ")

    for c in string.punctuation:
        s = s.replace(c, " ")
        s = s.replace("  ", " ")

    s = s.replace("     ", " ")
    s = s.replace("    ", " ")
    s = s.replace("   ", " ")
    s = s.replace("  ", " ")
    s = s.strip()

    return s

def check_abbrev_mult(list_3, str_2):

    str_2_tokens = str_2.split(" ")
    x = len(list_3)


    for str_1 in list_3:
        str_1_tokens = str_1.split(" ")

        if len(str_1_tokens) != len(str_2_tokens):
            x = x-1
            continue

        for token_1, token_2 in zip(str_1_tokens, str_2_tokens):
            res_1 = token_1.find(token_2)
            res_2 = token_2.find(token_1)

            if not (res_2==0 or res_1==0):
                x = x-1
                break
    if x == 0:
        return False

    return True

def check_abbrev_vectorized(vector_1, str0):
    output0 = np.array([check_abbrev_mult(i, str0) for i in vector_1])
    return output0

def check_abbrev_column(vector_1, str0):
    output0 = np.array([check_abbrev(i, str0) for i in vector_1])
    return output0

def match_source(arr_m, arr_s):
    return np.array([(elem0 in arr_s) for elem0 in arr_m])

def check_source(value):
    res = (sources[:, 2] == value) | (sources[:, 3] == value) | (sources[:, 4] == value)| (sources[:, 5] == value) | (sources[:, 6] == value)

    return res

def get_id_of_source(value):

    value_2 = clean_punc_from_str(value)
    res = sources[np.where(check_source(value_2))]
    if len(res)==0:
            res = sources[np.where((check_abbrev_vectorized(sources[:, 2:7], value_2) ) |
                           check_source(value_2) )]
    ids=[]

    for elem in res:
        ids.append(elem[0])

    if len(ids)>0:
        return  ids


    # check it with book names
    res = books[np.where((check_abbrev_vectorized(books[:, 1:3], value_2)) | (books[:, 1] == value_2) | (books[:, 2] == value_2)) ]

    for elem in res:
        ids.append(elem[0])

    return ids

def get_doi_val(values):
    for value in values:
        for token in identifier_tokens:
            if token in value:
                val = value.replace(token, "")
                val = val.replace(" ","")
                print("doi is:")
                print(val)
                print("********")
                return val
    return "None"

def convert_matrix_to_dict(year_index):
    
    X = len(WOS[year_index])
    
    author_name = {}
    title       = {}
    source      = {}
    volume      = {}
    issue       = {}
    page_b      = {}
    page_e      = {}
    number      = {}
    doi         = {}

    for i in range(0, X):

        var_id          = i
        var_name        = WOS[year_index][i][1]
        var_title       = WOS[year_index][i][3]
        var_source      = WOS[year_index][i][4]

        var_volume      = WOS[year_index][i][6]
        var_issue       = WOS[year_index][i][7]

        var_page_b      = WOS[year_index][i][8]
        var_page_e      = WOS[year_index][i][9]

        var_page_number = WOS[year_index][i][10]
        var_doi         = WOS[year_index][i][11]

        if var_name in author_name:
            author_name[var_name].append(var_id)
        else:
            author_name[var_name] = [var_id]


        if var_title in title:
            title[var_title].append(var_id)
        else:
            title[var_title] = [var_id]


        if var_source in source:
            source[var_source].append(var_id)
        else:
            source[var_source] = [var_id]        


        if var_volume in volume:
            volume[var_volume].append(var_id)
        else:
            volume[var_volume] = [var_id] 


        if var_issue in issue:
            issue[var_issue].append(var_id)
        else:
            issue[var_issue] = [var_id] 


        if var_page_b in page_b:
            page_b[var_page_b].append(var_id)
        else:
            page_b[var_page_b] = [var_id] 


        if var_page_e in page_e:
            page_e[var_page_e].append(var_id)
        else:
            page_e[var_page_e] = [var_id] 


        if var_page_number in number:
            number[var_page_number].append(var_id)
        else:
            number[var_page_number] = [var_id]         


        if var_doi in doi :
            doi[var_doi].append(var_id)
        else:
            doi[var_doi] = [var_id]  

    dict_data = {}
    dict_data['author']   = author_name
    dict_data['title']    = title
    dict_data['source']   = source 
    dict_data['volume']   = volume
    dict_data['issue']    = issue
    dict_data['page_b']   = page_b
    dict_data['page_e']   = page_e
    dict_data['number']   = number
    dict_data['doi']      = doi
    
    return dict_data  

def load_wos_data(WOS_PATH):
    for i in range(1980, 2024):
        path = WOS_PATH +str(i)+'.csv'
        print(i)
        WOS[i] = pd.read_csv(path, sep=';', header=0, usecols=["pub_id", "author_first", "author_et_al", "title", "source_id","book_id", "volume","issue","page_begin","page_end","article_no","doi"]

         ,dtype={"pub_id": int, "author_first": "string", "author_et_al": "string", "title": "string",
                 "source_id": "string", "book_id": "string", "volume": "string", "issue": "string",
                 "page_begin": "string", "page_end": "string","article_no":"string", "doi":"string"}).astype(str).to_numpy()

        WOS[i][:,1] = np.char.lower(WOS[i][:,1].astype("str"))
        WOS[i][:,1] = np.char.replace(WOS[i][:,1].astype("str"), "-", "")
        WOS[i][:,2] = np.char.lower(WOS[i][:,2].astype("str"))
        WOS[i][:,2] = np.char.replace(WOS[i][:,2].astype("str"), "-", "")
        # change the title of books format
        WOS[i][:,3] = clean_punc_from_str_list(WOS[i][:,3].astype("str"))
    return WOS

def load_sources(Source_PATH):
    Sources = pd.read_csv(Source_PATH, sep=';', header=0, usecols=["source_id","source_type_id","source_title","source_abbrev", "source_abbrev_iso", "source_abbrev_11", "source_abbrev_29", "issn_print", "issn_e", "publisher_unified", "is_conference_proceeding"]
                         ,dtype={"source_id": int, "source_type_id": int,
        "source_title": "string", "source_abbrev": "string", "source_abbrev_iso": "string",
        "source_abbrev_11": "string", "source_abbrev_29": "string", "issn_print":"string",
         "issn_e":"string", "publisher_unified":"string", "is_conference_proceeding":"string"}).astype(str).to_numpy()

    Sources[:,2] = clean_punc_from_str_list(Sources[:,2].astype("str"))
    Sources[:,3] = clean_punc_from_str_list(Sources[:,3].astype("str"))
    Sources[:,4] = clean_punc_from_str_list(Sources[:,4].astype("str"))
    Sources[:,5] = clean_punc_from_str_list(Sources[:,5].astype("str"))
    Sources[:,6] = clean_punc_from_str_list(Sources[:,6].astype("str"))

    return Sources


def load_books(PUB_PATH):
    books = pd.read_csv(PUB_PATH, sep=';', header=0, usecols=["book_id", "book_title", "book_title_translated", "isbn_print_1", "isbn_print_2", "isbn_e", "doi"]
                         ,dtype={"book_id": "string", "book_title": "string", "book_title_translated": "string", "isbn_print_1": "string",
        "isbn_print_2": "string", "isbn_e": "string", "doi":"string"}).astype(str).to_numpy()

    books[:,1] = clean_punc_from_str_list(books[:,1].astype("str"))
    books[:,2] = clean_punc_from_str_list(books[:,2].astype("str"))

    return books

def create_vector_of_ref(reference):
    
    author_first = "None"
    title = "None"
    source_id = "None"
    volume = "None"
    issue = "None"
    page_begin = "None"
    page_end = "None"
    article_no = "None"
    doi = "None"
    year = "None"

    for label in reference:
        if len(reference[label])>0:

            if label == "Year":
                year = reference["Year"][0]
                year = year.strip()

            elif label == "Auth":
                author_first = get_name_format_excel(reference["Auth"][0])

            elif label == "Title":
                title = clean_punc_from_str(reference["Title"][0])

            elif label == "Src":
                source_id = get_id_of_source(reference["Src"][0])


            elif label == "Vol":
                volume = reference["Vol"][0]
                volume = volume.strip()
                volume = volume.replace(" ","")

            elif label == "Issue":
                issue = reference["Issue"][0]
                issue = issue.strip()
                issue = issue.replace(" ","")

            elif label == "Page":
                page_begin, page_end = get_cleaned_pages(reference["Page"][0])

            elif label == "Num":
                article_no = reference["Num"][0]
                article_no = article_no.strip()
                article_no = article_no.replace(" ","")

            elif label == "Iden":
                doi = get_doi_val(reference["Iden"])


    vec = [year, author_first, "", title, source_id, volume, issue, page_begin, page_end, article_no, doi]
    
    return vec


def intersect(a, b):
    return list(set(a).intersection(b))

def intersect_fields(res):
    
    dict_out       = {}
    match_score    = 0
    max_count      = 0
    matched_entity = []
    index          = 0
    
    for key in res:
        index += 1
        # each key is doi, title, volume, source, ...        
        for row_index in res[key]:
            if row_index in dict_out:
                # each key is row index with array of above fields                 
                dict_out[row_index].append(index)
            else: 
                dict_out[row_index] = [index]
    
    for key in dict_out:
 
        if len(dict_out[key]) > match_score:
            match_score = len(dict_out[key])
            matched_entity = [key]

        elif len(dict_out[key]) == match_score:
            matched_entity.append(key)
                

    return dict_out, match_score, matched_entity

def find_matching_entry(vec):
    
    exact_match = False
    matched_index = -1
    title = ""
    res = {}
    res['doi']      = []
    res['title']    = []
    res['volume']   = []
    res['issue']    = []
    res['number']   = []
    res['source']   = []
    res['author']   = []
    res['page']     = []
    
    
    if vec[0].isdigit():
        year_index = int(vec[0])
        if (int(year_index) >= 1980) and (int(year_index) <= 2023):
            print(vec)

            
            #  doi           
            if (vec[10] in WOS_dict[year_index]['doi']):
                res['doi'] = WOS_dict[year_index]['doi'][vec[10]].copy()
                
            #  title
            if (vec[3] in WOS_dict[year_index]['title']):
                res['title'] = WOS_dict[year_index]['title'][vec[3]].copy()
                
            #  volume
            if (vec[5] in WOS_dict[year_index]['volume']):
                res['volume'] = WOS_dict[year_index]['volume'][vec[5]].copy()
                
            #  issue
            if (vec[6] in WOS_dict[year_index]['issue']):
                res['issue'] = WOS_dict[year_index]['issue'][vec[6]].copy()
                
            #  number
            if (vec[9] in WOS_dict[year_index]['number']):
                res['number'] = WOS_dict[year_index]['number'][vec[9]].copy()
            
            #  source  
            if vec[4] != 'None':
                for source_index in vec[4]:
                    if source_index in WOS_dict[year_index]['source']:
                        res['source'] += WOS_dict[year_index]['source'][source_index].copy()
                        
            
            # author
            if vec[1] != '' and vec[1] != 'None':
                for author_i in WOS_dict[year_index]['author']:
                    if (vec[1] in author_i) or (author_i in vec[1]):
                        res['author'] += WOS_dict[year_index]['author'][author_i].copy()

            
            #page
            if (vec[7] in WOS_dict[year_index]['page_b']):
                if vec[8] in ["NONE", "+", "&"]:
                    res['page'] = WOS_dict[year_index]['page_b'][vec[7]].copy()
                else:
                    if (vec[8] in WOS_dict[year_index]['page_e']):
                        res['page'] = intersect(WOS_dict[year_index]['page_b'][vec[7]], WOS_dict[year_index]['page_e'][vec[8]])


                    if "NONE" in WOS_dict[year_index]['page_e']:
                        res['page']  += intersect( WOS_dict[year_index]['page_b'][vec[7]], (WOS_dict[year_index]['page_e']["NONE"]))

                    if "+" in WOS_dict[year_index]['page_e']:
                        res['page'] += intersect(WOS_dict[year_index]['page_b'][vec[7]], (WOS_dict[year_index]['page_e']["+"]))

                    if "&" in WOS_dict[year_index]['page_e']:
                        res['page'] += intersect(WOS_dict[year_index]['page_b'][vec[7]], (WOS_dict[year_index]['page_e']["&"]))
    
                
            dict_final, match_score, matched_indices = intersect_fields(res)
            max_count = len(matched_indices)
            print(max_count)
            print(match_score)
            
            # matching process started
            if len(res['doi'])>0:
                print("exact match doi")
                exact_match = True
                matched_entity_index = res['doi'][0]
                match_score = len(dict_final[matched_entity_index])
                title = WOS[year_index][matched_entity_index][3]
                print(WOS[year_index][matched_entity_index])
                print(dict_final[matched_entity_index])
                print("<-***************************************->")
                return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            elif len(res['title'])>0:
                print("exact match title")
                exact_match = True
                matched_entity_index = res['title'][0]
                match_score = len(dict_final[matched_entity_index])
                title = WOS[year_index][matched_entity_index][3]
                print(WOS[year_index][matched_entity_index])
                print(dict_final[matched_entity_index])
                print("<-***************************************->")
                return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            elif max_count==1 and match_score>=2:
                print("exact match Rule 1")
                print("matched with score: ", match_score)
                print(matched_indices)
                exact_match = True
                matched_entity_index = matched_indices[0]
                title = WOS[year_index][matched_entity_index][3]
                print(WOS[year_index][matched_entity_index])
                print(dict_final[matched_entity_index])
                print("<-***************************************->")
                return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            elif match_score>=3:
                # if source volume and page are matched
                print("***********")                 
                for row in matched_indices:
                    if set([6,3,8]) <= set(dict_final[row]):
                        print("exact match Rule 2")
                        exact_match = True
                        matched_entity_index = row 
                        title = WOS[year_index][matched_entity_index][3]
                        print(WOS[year_index][matched_entity_index])
                        print(dict_final[matched_entity_index])
                        print("<-***************************************->")
                        return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            
            print("<-***************************************->")
            
            print("not matched: ")
            return match_score, False, "", None
            
            
        return -1, exact_match, title, None

    return -2, exact_match, title, None            



# *************** Parameters 

output_matching_path = 'matched_publications.pkl'
input_path_to_vectors_of_fields = 'fields_all_data.pkl'

WOS_PATH = "/WOS/pub"
Source_PATH = '/WOS/source2213.csv'
PUB_PATH = '/WOS/book2213.csv'

# ******************** Loading WOS dataset 


sources = load_sources(Source_PATH)
books = load_books(PUB_PATH)
WOS = load_wos_data(WOS_PATH)

WOS_dict = {}
for key in WOS:
    WOS_dict[key] = convert_matrix_to_dict(key)
    print(key)
    print("********")


# ******************** Load vectors


with open(input_path_to_vectors_of_fields, 'rb') as handle:
    vectors = pickle.load(handle)


new_vectors = []
index = 0
for vec in vectors:
    rep_vec = create_vector_of_ref(vec)
    new_vectors.append([vec["Ref"], rep_vec, vec["row_index"]])
    index+=1
    print('Progress is: ', (index/len(vectors))*100)    



# ******************** Start Matching Extracted fields    


outputs = []

for vec in new_vectors:
    a, b, c, d = find_matching_entry(vec[1])
    tmp = {'match_score': a,
    'is_matched': b,
    'title': c,
    'matched_publication': d,
    'vector': vec}
    outputs.append(tmp)


with open(output_matching_path, 'wb') as handle:
    pickle.dump(outputs, handle)    
