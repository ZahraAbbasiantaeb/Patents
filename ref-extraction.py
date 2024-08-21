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


label_list = ['O', 'B-PER', 'B-LOC']
dict1 = {'O':'O','B':'B-PER','I':'B-LOC'}
label_encoding_dict = {'O':0,'B-PER':1,'B-LOC':2}

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


def down_sample(input_seqs, labels):
    
    pos_input_seqs = []
    pos_labels = []
    
    neg_input_seqs = []
    neg_labels = []
    
    for i in range(0, len(labels)):
        if ('B-PER' in labels[i] or 'B-LOC' in labels[i]):
            pos_input_seqs.append(input_seqs[i])
            pos_labels.append(labels[i])
            
        else:
            neg_input_seqs.append(input_seqs[i])
            neg_labels.append(labels[i])            

    return pos_input_seqs, pos_labels

def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], 
            "f1": results["overall_f1"], "accuracy": results["overall_accuracy"],
           "prec_B":results["PER"]["precision"], "recall_B":results["PER"]["recall"], "B_count":results["PER"]["number"],
           "prec_I":results["LOC"]["precision"], "recall_I":results["LOC"]["recall"], "I_count":results["LOC"]["number"]}

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(list(examples["tokens"]), padding='max_length', truncation=True, max_length=512,
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

def make_dataset_optimized(corpus, tags):
  
    T_prime = []
    L_prime = []
    input_seqs = []
    labels = []
  
    tokenized_corpus = tokenizer(corpus, is_split_into_words=True, add_special_tokens=False)
    word_ids = tokenized_corpus.word_ids()

    token_length = 0
    cur_index = 0
    total_length = 0

    for elem in word_ids:
        if elem == cur_index:
            token_length += 1
    
        else:
            if (total_length + token_length) > (MAX_SEQ_LENGTH-2):
        # add new sequence to the list
                input_seqs.append(T_prime)
                labels.append(L_prime)
                T_prime = []
                L_prime = []
                total_length = 0
      
            T_prime.append(corpus[cur_index])
            L_prime.append(tags[cur_index])
            total_length+=token_length
            token_length = 1
            cur_index = elem

    T_prime.append(corpus[cur_index])
    L_prime.append(tags[cur_index])
    input_seqs.append(T_prime)
    labels.append(L_prime)

    return input_seqs, labels

def label_to_ner(label):
    return dict1[label]
  
def read_patents(patents_path):
    with (open(patents_path, "rb")) as openfile:
        patents = pickle.load(openfile)

    return patents

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

def convert_pred_labels_to_text(res):

    patent_num = res["patent_num"][0]
    refs = {}
    refs[patent_num] = []
    mode = "None"
    tmp = ""

    for i in range (0, len(res["patent_num"])):
        
        if patent_num!=res["patent_num"][i]:
            # new patent 
            if mode == "Inside":
                refs[patent_num].append(tmp)
            patent_num = res["patent_num"][i]
            refs[patent_num] = []
            mode = "None"
            tmp = ""        
        
        patent_num = res["patent_num"][i]
        words = res["words"][i]
        preds = res["tag"][i]
        labels = res["all_labels"][i][0]
        word_ids = res["word_ids"][i]
        curr_w_id = word_ids[0]
        
        for word, pred, label, word_id in zip(words, preds, labels, word_ids):
            
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
                if not word=="":
                    tmp += " " + word
            
            elif pred =="O":
                if mode == "Inside":
                    refs[patent_num].append(tmp)
                
                tmp = ""
                mode = "None"
    return refs

def return_sequences_V2_train(patents, test_patents):
    
    corpus = []
    tags = []

    input_seqs = []
    labels = []
    for type in ["EPO", "USPTO"]:
        for patent_number in patents[type]:
            if not patent_number in test_patents:
                tmp = patents[type][patent_number]

                corpus += (tmp[0])
                tags += [label_to_ner(elem) for elem in tmp[1]]

    X, Y = make_dataset_optimized(corpus, tags)
    input_seqs += (X)
    labels += (Y)
    
    return input_seqs, labels

def return_sequences_V3_test(patents, test_patents):
    
    corpus = []
    tags = []

    input_seqs = {}
    labels = {}
    for typ in ["EPO", "USPTO"]:
        for patent_number in patents[typ]:
            if patent_number in test_patents:
                
                tmp=patents[typ][patent_number]
                corpus=tmp[0]
                tags=[label_to_ner(elem) for elem in tmp[1]]

                X, Y = make_dataset_optimized(corpus, tags)
                input_seqs[patent_number]=X
                labels[patent_number]=Y
    
    return input_seqs, labels

def return_sequences_V2_test(patents, test_patents):
    
    corpus = []
    tags = []

    input_seqs = []
    labels = []
    for typ in ["EPO", "USPTO"]:
        for patent_number in patents[typ]:
            if patent_number in test_patents:
                
                tmp = patents[typ][patent_number]
                corpus += (tmp[0])
                tags += [label_to_ner(elem) for elem in tmp[1]]

    X, Y = make_dataset_optimized(corpus, tags)
    input_seqs += (X)
    labels += (Y)
    
    return input_seqs, labels            

  
                
task = "ner" # Should be one of "ner", "pos" or "chunk"
# model_checkpoint = "./bert-base-cased"
# model_checkpoint = "./bert-for-patents"
model_checkpoint = "allenai/scibert_scivocab_cased"
# model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
# model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"
# model_checkpoint = "./PatentBERT"

batch_size = 8
label_all_tokens = False
DOWN_SAMPLE = True
LOOCV = False
MAX_SEQ_LENGTH = 512
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=False)
TRAINING = True
PREDICTING = True


assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    num_train_epochs=6,
    weight_decay=1e-5,)

max_length = 512
data_collator = DataCollatorForTokenClassification(tokenizer, padding='max_length', max_length=512, 
                                                    label_pad_token_id=-100)

metric = load_metric("seqeval")

output_model_path = 'Models/ref-extraction.model'
patents_path = "datasets/Ref_extraction_with_reflist_V1.pickle"
prediction_path = 'output/predicted_references.pkl'

# Load Dataset

patents = read_patents(patents_path)


# Train dataset
print("Reading train data ...")
input_seqs_train, labels_train = return_sequences_V2_train(patents, test_patents)

if DOWN_SAMPLE == True:
    print("Down samplin train data ...")
    input_seqs_train, labels_train = down_sample(input_seqs_train, labels_train)
pd_train = pd.DataFrame([input_seqs_train, labels_train], index=['tokens', 'ner_tags']).transpose()

# Test dataset
print("Read test data ... ")
input_seqs_test, labels_test = return_sequences_V2_test(patents, test_patents)
pd_test = pd.DataFrame([input_seqs_test, labels_test], index=['tokens', 'ner_tags']).transpose()


print("Generating pandas dataset ...")
train_dataset = Dataset.from_pandas(pd_train)
test_dataset = Dataset.from_pandas(pd_test)

print("Tokenizing pandas dataset ...")
tokenized_patent_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_patent_test = test_dataset.map(tokenize_and_align_labels, batched=True)


print("Training the model  ...")


if TRAINING:

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    trainer = Trainer(
        model,
        args,
        train_dataset = tokenized_patent_train,
        eval_dataset = tokenized_patent_train,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics)

    print("Start training ...")
    trainer.train()    
    trainer.evaluate()
    trainer.save_model(output_model_path)

if PREDICTING:

    model = AutoModelForTokenClassification.from_pretrained(output_model_path, num_labels=len(label_list))
    test_preds = []
    test_tokens = []
    patent_num = []
    all_labels = []
    id_of_tokens = []
    res={}
    index=1

    for patent in input_seqs_test:

        for seq, label in zip(input_seqs_test[patent], labels_test[patent]):
        
            tokens = tokenizer(seq, padding='max_length', truncation=True, max_length=512, is_split_into_words=True)
            torch.tensor(tokens['input_ids']).unsqueeze(0).size()
            preds = model.forward(input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
            preds = torch.argmax(preds.logits.squeeze(), axis=1)
            token_ids = tokens.word_ids()
            aligned_labels = get_aligned_labels(tokens, label)
            predictions = [label_list_2[i] for i in preds]
            words = tokenizer.batch_decode(tokens['input_ids'])
            all_labels.append(aligned_labels)
            test_preds.append(predictions)
            test_tokens.append(words)
            patent_num.append(patent)
            id_of_tokens.append(token_ids)
            
            print(index)
            index+=1
            
            
    res["patent_num"] = patent_num        
    res["words"] = test_tokens
    res["tag"] = test_preds
    res["all_labels"] = all_labels
    res["word_ids"] = id_of_tokens
    refs = convert_pred_labels_to_text(res)

    with open(prediction_path, 'wb') as handle:
        pickle.dump(refs, handle)

