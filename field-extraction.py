from sklearn.model_selection import train_test_split
from datasets.features.features import ClassLabel
import pandas as pd
import numpy as np
from datasets import Dataset
import datasets
import transformers
import pickle
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch




label_list = ['O', 'B-PER', 'B-EVE', 'B-NAT', 'B-ART', 'B-GPE', 'B-GEO', 'B-TIM', 
              'B-ORG', 'B-LOC', 'I-PER','I-EVE', 'I-NAT', 'I-ART', 'I-GPE', 'I-GEO']

label_list_2 = ['O', 'Auth-B', 'Year-B', 'Src-B', 'Ttl-B', 'Page-B', 'Iden-B','Vol-B', 'Issue-B','Num-B',
                'Auth-I','Year-I','Src-I','Ttl-I', 'Page-I', 'Iden-I']

label_encoding_dict = {'O':0,'B-PER':1,'B-EVE':2, 'B-NAT':3, 'B-ART':4, 'B-GPE':5,  'B-GEO':6, 
                        'B-TIM':7, 'B-ORG':8, 'B-LOC':9,'I-PER':10, 'I-EVE':11, 'I-NAT':12,'I-ART':13,
                        'I-GPE':14, 'I-GEO':15}

test_patents=['3016308', '07185039 ', '0265010', '07130023 ', '07608978 ', '0106404', '2376427', '0646876', 
              '0687091', '0750621', '0326940', '06531583 ', '08129103 ', '3419289', '0323989', '0836178', 
              '06797868 ', '07672040 ', '10389262 ', '1433297', '0506071', '2610256', '0177558', '2618134', 
              '06907831 ', '2223240', '2794838', '1200612', '06916331 ', '1100819', '2049570', '0730652', 
              '07305021 ', '2137307', '1032367', '07759652 ', '08656455 ', '0627701', '3407862', '09138162 ', 
              '3079274', '2243372', '06946517 ', '07169418 ', '1501090', '1149076', '06876309 ', '11299445 ', 
              '08886538 ', '2275863', '0686172', '08828792 ', '08735377 ', '3232619', '08188070 ', '1059934', 
              '0646600', '07498164 ', '0540264', '07807616 ', '10577471 ', '3489012', '0502599', '0333216', 
              '0723436', '2510476', '1462455', '0532954', '11281977 ', '0190740', '0540611', '1596226', '09225040 ', 
              '2705706', '3154983', '0720594', '07813983 ', '07230060 ', '1933430', '09284263 ', '2081911', 
              '09088047 ', '07058873 ', '07578285 ', '08362137 ', '0848880', '08102492 ', '08012611 ', '07687522 ', 
              '09709723 ', '09663433 ', '2015854', '2620303', '2989157', '07175994 ', '07868210 ', '07491649 ',
              '09143150 ', '0485494', '0616633', '06818333 ', '0159714', '0456304', '09428501 ', '3126444', '2663662',
              '11149014 ', '08788755 ', '1552058', '2544593', '0651331', '1145096', '07280406 ', '1452551', 
              '0430096', '09007551 ', '09051337 ', '1328509', '1172696', '07208086 ', '2262905', '07438924 ', 
              '07791059 ', '0178912', '0650286', '07378383 ', '06626122 ', '08101613 ', '0731385', '1695338',
              '08382986 ', '1244267', '2812040', '1233058', '2357464', '11301770 ', '10612079 ', '1307768',
              '0078320', '07393867 ', '1173059', '07328059 ', '09206217 ', '1934823']

dict1 = {'O':'O',
     'Auth-B':'B-PER',
     'Year-B':'B-EVE', 
     'Src-B':'B-NAT', 
     'Ttl-B':'B-ART', 
     'Page-B':'B-GPE', 
     'Iden-B':'B-GEO', 
     'Vol-B':'B-TIM', 
     'Issue-B':'B-ORG', 
     'Num-B':'B-LOC',
     'Auth-I':'I-PER', 
     'Year-I':'I-EVE', 
     'Src-I':'I-NAT', 
     'Ttl-I':'I-ART', 
     'Page-I':'I-GPE', 
     'Iden-I':'I-GEO'}
    
def label_to_ner(label):
    return dict1[label]

def return_sequences(path):
    
    with open(path, "rb") as openfile:
        data = pickle.load(openfile)
    
    input_seqs_train = []
    labels_train = []

    input_seqs_test = []
    labels_test = []
    
    for elem in data:
        if elem[3] in test_patents:
            input_seqs_test.append(elem[1])
            labels_test.append([label_to_ner(x) for x in elem[2]])
        else:
            input_seqs_train.append(elem[1])
            labels_train.append([label_to_ner(x) for x in elem[2]])
    
    return input_seqs_train, input_seqs_test, labels_train, labels_test

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(list(examples["tokens"]), padding='max_length', truncation=True, max_length=max_length_for_seq,
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs

def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], 
          "f1": results["overall_f1"], "accuracy": results["overall_accuracy"],
          "prec_Auth":results["PER"]["precision"], "recall_Auth":results["PER"]["recall"], "Auth_count":results["PER"]["number"],
          "prec_Num":results["LOC"]["precision"], "recall_Num":results["LOC"]["recall"], "Num_count":results["LOC"]["number"],
          "prec_Year":results["EVE"]["precision"], "recall_Year":results["EVE"]["recall"], "Year_count":results["EVE"]["number"],
          "prec_Src":results["NAT"]["precision"], "recall_Src":results["NAT"]["recall"], "Src_count":results["NAT"]["number"],
          "prec_Ttl":results["ART"]["precision"], "recall_Ttl":results["ART"]["recall"], "Ttl_count":results["ART"]["number"],
          "prec_Page":results["GPE"]["precision"], "recall_Page":results["GPE"]["recall"], "Page_count":results["GPE"]["number"],
          "prec_Iden":results["GEO"]["precision"], "recall_Iden":results["GEO"]["recall"], "Iden_count":results["GEO"]["number"],
          "prec_VOL":results["TIM"]["precision"], "recall_VOL":results["TIM"]["recall"], "VOL_count":results["TIM"]["number"],
          "prec_Issue":results["ORG"]["precision"], "recall_Issue":results["ORG"]["recall"], "Issue_count":results["ORG"]["number"]
           }
 

 task = "ner" # Should be one of "ner", "pos" or "chunk"

def get_aligned_labels(tokens, label):
    
    word_ids = tokens.word_ids()
    previous_word_idx = None
    label_ids = []
    labels=[]

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif label[word_idx] == '0':
            label_ids.append(0)
        elif word_idx != previous_word_idx:
            label_ids.append(label_encoding_dict[label[word_idx]])
        else:
            label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
        previous_word_idx = word_idx
    labels.append(label_ids)
    
    return labels

def return_sequences_inference(path):
    
    with open(path, "rb") as openfile:
        data = pickle.load(openfile)

    input_seqs = []
    for patent in data:
        for elem in data[patent]:
            input_seqs.append(elem)
    
    return input_seqs

def convert_inference_output_to_fields(test_preds, test_tokens, word_ids):
    X = 0
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

        



model_checkpoint = "allenai/scibert_scivocab_cased"
# model_checkpoint = "./bert-base-cased"
# model_checkpoint = "./dslim/bert-base-NER"
# model_checkpoint = "./bert-for-patents"
# model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
# model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"
# model_checkpoint = "./PatentBERT"

batch_size = 16
label_all_tokens = False
max_length_for_seq = 200
MAX_SEQ_LENGTH = max_length_for_seq
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=False)
extracted_references_path = "./datasets/FED_VL.pickle"
output_model_path = 'Models/field-extraction-complete.model'
extracted_fields_path ='Output/extracted_fields_test.pickle'  


assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    weight_decay=1e-5,)

max_length = max_length_for_seq
data_collator = DataCollatorForTokenClassification(tokenizer, padding='max_length', max_length=max_length_for_seq, 
                                                    label_pad_token_id=-100)
metric = load_metric("seqeval")



print("Started")

# Train dataset
input_seqs_train, input_seqs_test, labels_train, labels_test = return_sequences(extracted_references_path)

pd_train = pd.DataFrame([input_seqs_train, labels_train], index=['tokens', 'ner_tags']).transpose()
print("Train length is ...")
print(len(labels_train))

# Test dataset
print("Test length is ...")
print(len(labels_test))
pd_test = pd.DataFrame([input_seqs_test, labels_test], index=['tokens', 'ner_tags']).transpose()


print("Generating pandas dataset ...")
train_dataset = Dataset.from_pandas(pd_train)
test_dataset = Dataset.from_pandas(pd_test)

print("Tokenizing pandas dataset ...")
tokenized_patent_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_patent_test = test_dataset.map(tokenize_and_align_labels, batched=True)


if TRAINING == True:
    print("Training the model ...")

        
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_patent_train,
        eval_dataset=tokenized_patent_train,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    print("Start training ...")
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_model_path)

if PREDICTING:

    model = AutoModelForTokenClassification.from_pretrained(output_model_path, num_labels=len(label_list))
    all_labels = []
    test_preds = []
    test_tokens = []
    word_ids = []

    for seq_1 in input_seqs_test:
        seq = [elem for elem in seq_1.split(" ")]
        tokens = tokenizer(seq, padding='max_length', truncation=True, max_length=max_length_for_seq, is_split_into_words=True)
        torch.tensor(tokens['input_ids']).unsqueeze(0).size()
        preds = model.forward(input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
        preds = torch.argmax(preds.logits.squeeze(), axis=1)
        id_of_words = tokens.word_ids()
        
        predictions = [label_list_2[i] for i in preds]
        words = tokenizer.batch_decode(tokens['input_ids'])

        word_ids.append(id_of_words)
        test_preds.append(predictions)
        test_tokens.append(words)
        all_results = convert_inference_output_to_fields(test_preds, test_tokens, word_ids)

        with open(extracted_fields_path, 'wb') as handle:
            pickle.dump(all_results, handle)
          