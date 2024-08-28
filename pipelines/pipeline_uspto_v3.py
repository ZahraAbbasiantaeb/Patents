from datasets.features.features import ClassLabel
from datasets import Dataset
import datasets
import itertools
import pandas as pd
import numpy as np
import random
import pickle
import transformers
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
import string
import sqlite3
from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
import gc


# *********************************


file_header='PATN'
xml_header='<?xml version="1.0" encoding="UTF-8"?>\n'
label_list_RE = ['O', 'B-PER', 'B-LOC']
label_list_2_RE = ['O', 'B', 'I']
label_list_FE = ['O', 'B-PER', 'B-EVE', 'B-NAT', 'B-ART', 'B-GPE', 'B-GEO', 'B-TIM',
              'B-ORG', 'B-LOC', 'I-PER','I-EVE', 'I-NAT', 'I-ART', 'I-GPE', 'I-GEO']

label_list_2_FE = ['O', 'Auth-B', 'Year-B', 'Src-B', 'Ttl-B', 'Page-B', 'Iden-B', 'Vol-B', 'Issue-B', 'Num-B',
                'Auth-I','Year-I','Src-I','Ttl-I', 'Page-I', 'Iden-I']

arr_page_seperator = ["–", "--", "- -", "-", "_", ":", "to", ",", "To", "and", "through", "And", "Through"]
identifier_tokens = ["doi :", "DOI :","doi=", "doi "]

# ************************************


def clean_text(entry_text):
     
    soup = BeautifulSoup(entry_text)
    paragraphs = soup.find_all('p')
    total_text = ''

    if (len(paragraphs)>0):
        for text in paragraphs:
            total_text += text.getText()+' '
    else:
        return 
    
    text_clean = pre_process(total_text)
    
    return text_clean

 
def pre_process(text):
    
    puncs = string.punctuation
    text = text.strip()
    
    for punc in puncs:
        text = text.replace(punc, " "+punc+ " ")
    
    text = text.replace("    ", " ")    
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    
    return text


def make_dataset_optimized_RE(corpus):
    corpus = clean_text(corpus)
    corpus = corpus.split(" ")

    tokenized_corpus = tokenizer_RE(corpus, is_split_into_words=True, add_special_tokens=False)
    word_ids = tokenized_corpus.word_ids()

    token_length = 0
    cur_index = 0
    total_length = 0
    T_prime = []
    input_seqs = []
    labels = []

    for elem in word_ids:
        if elem == cur_index:
            token_length += 1

        else:
            if (total_length + token_length) > (MAX_SEQ_LENGTH_RE - 2):
                # add new sequence to the list
                input_seqs.append(T_prime)
                T_prime = []
                total_length = 0

            T_prime.append(corpus[cur_index])
            total_length += token_length
            token_length = 1
            cur_index = elem

    T_prime.append(corpus[cur_index])
    input_seqs.append(T_prime)

    return input_seqs

def get_str_refs_from_pred_RE(words_list, preds_list, word_ids_list):
    extracted_refs = []

    for words, preds, word_ids in zip(words_list, preds_list, word_ids_list):
        tmp = ""
        curr_w_id = -1000
        mode = "None"
        for word, pred, word_id in zip(words, preds, word_ids):
            if word in ['[CLS]', '[PAD]', '[SEP]']:
                continue

            if word_id == curr_w_id:
                if mode == "Inside":
                    tmp += word
                continue

            curr_w_id = word_id

            if pred == "B":
                mode = "Inside"
                tmp = word

            elif pred == "I":
                if not word == "":
                    tmp += " " + word

            elif pred == "O":
                if mode == "Inside":
                    extracted_refs.append(tmp)
                tmp = ""
                mode = "None"

        if mode == "Inside":
            extracted_refs.append(tmp)

    return extracted_refs

def predict_refs_of_seq_RE(seq):
    tokens = tokenizer_RE(seq, padding='max_length', truncation=True, max_length=512, is_split_into_words=True)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()
    preds = model_RE.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0).to(device),
                             attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(device))
    preds = torch.argmax(preds.logits.squeeze(), axis=1)
    token_ids = tokens.word_ids()
    predictions = [label_list_2_RE[i] for i in preds]
    words = tokenizer_RE.batch_decode(tokens['input_ids'])

    return words, predictions, token_ids

def extract_refs_of_patent_RE(corpus):
    input_seqs = make_dataset_optimized_RE(corpus)
    words = []
    predictions = []
    token_ids = []

    for seq in input_seqs:
        words_1, predictions_1, token_ids_1 = predict_refs_of_seq_RE(seq)
        words.append(words_1)
        predictions.append(predictions_1)
        token_ids.append(token_ids_1)

    extracted_refs = get_str_refs_from_pred_RE(words, predictions, token_ids)

    return extracted_refs


def inference_FE(input_seqs):
    all_labels = []
    test_preds = []
    test_tokens = []
    word_ids = []

    for seq_1 in input_seqs:
        seq = [elem for elem in seq_1.split(" ")]
        tokens = tokenizer_FE(seq, padding='max_length', truncation=True, max_length=max_length_for_seq_FE,
                              is_split_into_words=True)
        torch.tensor(tokens['input_ids']).unsqueeze(0).size()
        preds = model_FE.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0).to(device),
                                 attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(device))
        preds = torch.argmax(preds.logits.squeeze(), axis=1)
        id_of_words = tokens.word_ids()

        predictions = [label_list_2_FE[i] for i in preds]
        words = tokenizer_FE.batch_decode(tokens['input_ids'])

        word_ids.append(id_of_words)
        test_preds.append(predictions)
        test_tokens.append(words)

    return word_ids, test_preds, test_tokens

def get_str_of_fields_FE(word_ids, test_preds, test_tokens):
    all_results = []
    for pred, tokens, ids_word in zip(test_preds, test_tokens, word_ids):

        flag = "None"
        tmp = ""
        res = {}
        txt = ""
        res["Ref"] = []
        res["Title"] = []
        res["Auth"] = []
        res["Page"] = []
        res["Src"] = []
        res["Iden"] = []
        res["Year"] = []
        res["Vol"] = []
        res["Issue"] = []
        res["Num"] = []
        prev_id = "None"

        for elem_pred, elem_token, w_id in zip(pred, tokens, ids_word):

            if elem_token == "[CLS]":
                continue

            if elem_token == "[SEP]" or elem_token == "[PAD]":

                res["Ref"] = txt.strip()
                if flag!="None":
                    res[flag].append(tmp.strip())
                    tmp = ""
                    flag = "None"
                break

            if w_id == prev_id:
                if flag!="None":
                    tmp += elem_token
                txt += elem_token
                continue

            prev_id = w_id
            txt += " "+ elem_token

            if elem_pred == "O":
                if flag!="None":
                    res[flag].append(tmp.strip())
                    tmp = ""
                    flag = "None"

            elif elem_pred == "Ttl-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Title"
                tmp = " " + elem_token

            if elem_pred == "Ttl-I":
                tmp += " " + elem_token

            elif elem_pred == "Auth-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Auth"
                tmp = " " + elem_token

            elif elem_pred == "Auth-I":
                tmp += " " + elem_token

            elif elem_pred == "Page-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Page"
                tmp = " " + elem_token

            elif elem_pred == "Page-I":
                tmp += " " + elem_token

            elif elem_pred == "Src-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Src"
                tmp = " " + elem_token

            elif elem_pred == "Src-I":
                tmp += " " + elem_token

            elif elem_pred == "Iden-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Iden"
                tmp = " " + elem_token

            elif elem_pred == "Iden-I":
                tmp += " " + elem_token

            elif elem_pred == "Year-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Year"
                tmp = " " + elem_token

            elif elem_pred == "Year-I":
                tmp += " " + elem_token

            elif elem_pred == "Vol-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Vol"
                tmp = elem_token.strip()

            elif elem_pred == "Issue-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Issue"
                tmp = elem_token.strip()

            elif elem_pred == "Num-B":
                if flag!="None":
                    res[flag].append(tmp.strip())
                flag = "Num"
                tmp = elem_token.strip()
        all_results.append(res)
    return all_results


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

def modify_page(b_page, e_page):

    n=len(e_page)
    t=len(b_page)
    page=b_page[0:t-n]+e_page

    return page

def get_name_format_excel(name_str):

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

    for punc in arr_page_seperator:
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

    # print("**********")
    # print("Error in page")
    # print(tmp)
    # print(value)

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

                return val
    return "None"

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


def find_matching_entry_v2(vec):
    
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

            
            # matching process started
            if len(res['doi'])>0:

                exact_match = True
                matched_entity_index = res['doi'][0]
                match_score = len(dict_final[matched_entity_index])
                title = WOS[year_index][matched_entity_index][3]
                return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            elif len(res['title'])>0:
                exact_match = True
                matched_entity_index = res['title'][0]
                match_score = len(dict_final[matched_entity_index])
                title = WOS[year_index][matched_entity_index][3]
                return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            elif max_count==1 and match_score>=2:
                exact_match = True
                matched_entity_index = matched_indices[0]
                title = WOS[year_index][matched_entity_index][3]

                return match_score, exact_match, title, WOS[year_index][matched_entity_index]
            
            elif match_score>=3:
                
                for row in matched_indices:
                    if set([6,3,8]) <= set(dict_final[row]):
                        exact_match = True
                        matched_entity_index = row 
                        title = WOS[year_index][matched_entity_index][3]

                        return match_score, exact_match, title, WOS[year_index][matched_entity_index]
                

            return match_score, False, "", None
            
            
        return -1, exact_match, title, None

    return -2, exact_match, title, None            


def find_matching_entry(vec):
    exact_match = False
    title = ""
    
    if vec[0].isdigit():
        if int(vec[0])>=1980 and int(vec[0])<=2023:
            matrix = WOS[int(vec[0])]
            # ------ compare (exact match) ---------
            col_bool = np.zeros([matrix.shape[0], 9], dtype=bool)

            # author name
            col_bool[:, 0] = (substr(matrix[:, 1], vec[1]) if vec[1] != '' and vec[1] != 'None' else False)

            #title
            col_bool[:, 1] = matrix[:, 3] == vec[3]

            # Source
            col_bool[:, 2] = match_source(matrix[:, 4], vec[4]) if vec[4] != 'None' else False

            # Volume
            col_bool[:, 3] = matrix[:, 6] == vec[5]

            # Issue
            col_bool[:, 4] = matrix[:, 7] == vec[6]

            # page
            col_bool[:, 5] = ((matrix[:, 8] == vec[7]) & (matrix[:, 9] == vec[8])) |  \
                             ((matrix[:, 8] == vec[7]) &
                              ( (vec[8] in ["NONE", "+", "&"]) |
                                ((matrix[:, 9] == "NONE") |
                                 (matrix[:, 9] == "+") | (matrix[:, 9] == "&") )))
            # Number
            col_bool[:, 6] = matrix[:, 10] == vec[9]

            # DOI
            col_bool[:, 7] = matrix[:, 11] == vec[10]

            # -------- matching formula ---------

            # Matching exception
            complete_match_cond = np.where((col_bool[:,1]==True) | (col_bool[:,7]==True))
            match_probability = np.sum(col_bool, axis=1)

            if len(complete_match_cond[0])>0:
                best_match_ind = complete_match_cond[0][0]
                exact_match = True
                title = matrix[best_match_ind,3]
                return match_probability[best_match_ind], exact_match, title, matrix[best_match_ind,:]

            #     General conditions
            else:
                # select columns with "match_probability"
                best_match_ind = np.argmax(match_probability)

            match_score = match_probability[best_match_ind]

            if match_score >= 1:
                matched_entries = np.where(match_probability==match_score)

                if ( len(matched_entries[0])==1 and match_score >= 2):
                    exact_match = True
                    title = matrix[best_match_ind,3]


                elif (match_score >= 3):
                    for entry in matched_entries[0]:
                        matched_fields = np.where(col_bool[entry,:]==True)
                        if (2 in matched_fields[0]) and (3 in matched_fields[0]) and (5 in matched_fields[0]):
                            exact_match = True
                            best_match_ind = entry
                            title = matrix[best_match_ind,3]
                            break

            del col_bool
            gc.collect()
            
            return match_probability[best_match_ind], exact_match, title, matrix[best_match_ind,:]
        
        return -1, exact_match, title, None

    return -2, exact_match, title, None

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
        if len(reference[label]) > 0:

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
                volume = volume.replace(" ", "")

            elif label == "Issue":
                issue = reference["Issue"][0]
                issue = issue.strip()
                issue = issue.replace(" ", "")

            elif label == "Page":
                page_begin, page_end = get_cleaned_pages(reference["Page"][0])

            elif label == "Num":
                article_no = reference["Num"][0]
                article_no = article_no.strip()
                article_no = article_no.replace(" ", "")

            elif label == "Iden":
                doi = get_doi_val(reference["Iden"])

    vec = [year, author_first, "", title, source_id, volume, issue, page_begin, page_end, article_no, doi]

    return vec

def match_references(vectors_fields_or_ref):
    
    new_vectors = []
    for vec in vectors_fields_or_ref:
        rep_vec = create_vector_of_ref(vec)
        new_vectors.append([vec["Ref"], rep_vec])

    match_score = []
    exact_match = []
    matched_entry = []

    for vec in new_vectors:
        a, b, c, d = find_matching_entry_v2(vec[1])
        match_score.append(a)
        exact_match.append(b)
        matched_entry.append(d)
    
    return match_score, exact_match, matched_entry


def insert_results_to_sql(patent_id, reference, fields_vec, match_score, exact_match, matched_entry):
    
    if exact_match==True:
        matched_pub = matched_entry[0]
    else:
        matched_pub=""
    
    if len(fields_vec["Auth"])>0:
        author = fields_vec["Auth"][0]
    else:
        author = ""
    
    if len(fields_vec["Src"])>0:
        source = fields_vec["Src"][0]
    else:
        source = ""

    if len(fields_vec["Year"])>0:
        year = fields_vec["Year"][0]
    else:
        year = ""
    
    if len(fields_vec["Title"])>0:
        title = fields_vec["Title"][0]
    else:
        title = ""        
    
    if len(fields_vec["Page"])>0:
        page = fields_vec["Page"][0]
    else:
        page = ""
    
    if len(fields_vec["Vol"])>0:
        volume = fields_vec["Vol"][0]
    else:
        volume = ""

    if len(fields_vec["Issue"])>0:
        issue = fields_vec["Issue"][0]
    else:
        issue = ""
    
    if len(fields_vec["Num"])>0:
        no = fields_vec["Num"][0]
    else:
        no = ""     
    
    if len(fields_vec["Iden"])>0:
        identifier = fields_vec["Iden"][0]
    else:
        identifier = ""   
    
    sqlite_insert_with_param = """INSERT INTO uspto_table
    (patent_id, reference, match_score, matched_pub, author, source, year, title, page, volume, issue, no, identifier) 
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    
    data_tuple = (patent_id, reference, str(match_score), matched_pub, author, source, year, title, page, volume, issue, no, identifier)
    write_bool = False
    for i in range(1000):
        try:
            cursor.execute(sqlite_insert_with_param, data_tuple)
            connection.commit()
            write_bool = True
            break
        except:
            time.sleep(0.01)

    if not write_bool:
        with open('not_writtent_sql.txt', 'a') as f:
            line = '\t'.join(
                [str(patent_id), reference, str(match_score), str(matched_pub), author, source, str(year), title, page,
                 str(volume), str(issue), str(no), identifier]) + '\n'
            f.write(line)
    
    return


def run_pipeline_on_patent(corpus, patent_number):
    
    extracted_refs = extract_refs_of_patent_RE(corpus)
    
    word_ids, test_preds, test_tokens = inference_FE(extracted_refs)

    fields_vectors = get_str_of_fields_FE(word_ids, test_preds, test_tokens)

    match_score, exact_match, matched_entry = match_references(fields_vectors)

    for i in range(0, len(extracted_refs)):
        insert_results_to_sql(patent_number, extracted_refs[i], fields_vectors[i], match_score[i], exact_match[i], matched_entry[i])
        
    return


def read_file(data_path):
    lines=[]
    with open(data_path, 'r', encoding = "ISO-8859-1") as f:
        data = f.read()
    patents=data.split(file_header)
    patents=patents[1:]
    return patents


def read_all_patents(mypath):
    all_patents = []
    file_list= [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in file_list:
        all_patents += read_file(file)  
    return all_patents


def get_utility_patents(all_patents):
    utility_patents={}
    
    for pat in all_patents:
        rows=pat.split('\n') 
        for row in rows:
            tmp=row.split('  ')
            if tmp[0]=='WKU':
                patent_id=tmp[1]
                if patent_id.isdigit() and len(patent_id) in [6, 7, 8, 9]:
                    utility_patents[patent_id] = pat
                    
    return utility_patents



def parse_V3_USPTO(patent):
    
    rows=patent.split('\n') 

    for row in rows:
        tmp=row.split('  ')
        if tmp[0]=='WKU':
            patent_id=tmp[1]
        if tmp[0]=='TTL':
            patent_title=tmp[1]
        
    
    patent_desc=''
    
    if 'CLMS' in patent:
        text = patent.split('CLMS')[0]
    else:
        text = patent
        
    lines=text.split('\n')
    write=False
    for line in lines:
        if line.startswith('      '):
            if write:
                patent_desc+=' '+line.strip()+' '
        if line.startswith('PA'):
            write=True
            if len(line.split('  '))>1:
                patent_desc+='\n'+line.split('  ')[1]
        else:
            write=False
    

    output_text=patent_id+": "+patent_title+" \n" + patent_desc
    return patent_id, output_text



WOS_PATH =  "/WOS/pub"
Source_PATH =  '/WOS/source2213.csv'
PUB_PATH =  '/WOS/book2213.csv'
USPTO_DATA_DIR_PATH = '/uspto/'
model_checkpoint_RE = "Models/ref-extraction-complete.model"
saved_model_checkpoint_FE = '/Models/field-extraction-complete.model'
USPTO_DB_PATH = '/USPTO-v3.db'



# *********************************
print("Loading REf-extraction model")
MAX_SEQ_LENGTH_RE = 512
tokenizer_RE = AutoTokenizer.from_pretrained(model_checkpoint_RE, do_lower_case=False)
assert isinstance(tokenizer_RE, transformers.PreTrainedTokenizerFast)
model_RE = AutoModelForTokenClassification.from_pretrained(model_checkpoint_RE, num_labels=len(label_list_RE)).to(device)


# *********************************
print("Loading Field-extraction model")
max_length_for_seq_FE = 200
tokenizer_FE = AutoTokenizer.from_pretrained(saved_model_checkpoint_FE, do_lower_case=False)
assert isinstance(tokenizer_FE, transformers.PreTrainedTokenizerFast)
model_FE = AutoModelForTokenClassification.from_pretrained(saved_model_checkpoint_FE, num_labels=len(label_list_FE)).to(device)


# *********************************

sources = load_sources(Source_PATH)
books = load_books(PUB_PATH)
WOS = load_wos_data(WOS_PATH)  

# *********************************


connection = sqlite3.connect(USPTO_DB_PATH)
cursor = connection.cursor()

years = [ 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990]

for year in years:
    mypath = USPTO_DATA_DIR_PATH+str(year)+'/'
    
    print("Reading files ...")
    all_patents = read_all_patents(mypath)
    
    print("Get utility patents ...")
    utility_patents = get_utility_patents(all_patents)
    
    patents_of_year = len(utility_patents)
    index = 0

    for key in utility_patents:
        index += 1
        patent_number, patent_text = parse_V3_USPTO(utility_patents[key])
        print("patents of file started: ")
        print(mypath)
        print(key)

        if len(patent_text)>0:
            run_pipeline_on_patent(patent_text, patent_number)
            print("done")
        print("progress of patents:", index, "/", patents_of_year) 

        
            
