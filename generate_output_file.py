import csv
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, AdamW, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datasets import load_metric
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from seqeval.metrics import classification_report, f1_score, accuracy_score

label_list=["B-Alteration","B-edit_R","B-false_R","B-filler_R","B-pet_R","B-repair_R","B-repeat_R","I-Alteration","I-edit_R","I-false_R","I-filler_R","I-pet_R","I-repair_R","I-repeat_R","O"]


id_to_label = {i : label_list[i] for i in range(len(label_list))}
label_to_id = {label_list[i] : i for i in range(len(label_list))}

index_to_label_dict={}
for i in range(len(label_list)):
    st = "LABEL_"+str(i)
    index_to_label_dict[st] = label_list[i]

sents=[[]]
# Open the TSV file in read mode
with open('./Test-Blind/Telugu/telugu_test_blind.tsv', encoding='utf-8') as tsvfile:
    # Use the Tab as the delimiter
    reader = csv.reader(tsvfile, delimiter='\t')
    # print(reader)

    # # Iterate through rows in the file
    for row in reader:
        # Process each row as needed
        if row[0] == '':
            sents.append([])
            continue
        sents[len(sents)-1].append(row[0])
        # print(row)  # Or do something else with the row data
    # print(i for i in sents)
#    for i in sents:
#        print (i)

chosen_model="./Telugu_model_2/"
model = AutoModelForTokenClassification.from_pretrained(chosen_model, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(chosen_model)


def predict_labels_for_sentences(model, tokenizer, sentences):
    predicted_labels_for_all_sents = []
    with torch.no_grad():
        for index, sentence in enumerate(sentences):
            input_tensors = tokenizer(
                sentence, is_split_into_words=True,truncation=True, return_tensors='pt', max_length=128)
            outputs = model(**input_tensors)
            logit_values = outputs.logits
            arg_max_torch = torch.argmax(logit_values, axis=-1)
            predicted_tokens_classes = [
                model.config.id2label[t.item()] for t in arg_max_torch[0]]
            word_ids = input_tensors.word_ids()
            previous_word_idx = 0
            label_ids = []
            for word_index in range(len(word_ids)):
                if word_ids[word_index] == None:
                    previous_word_idx = word_ids[word_index]
                elif word_ids[word_index] != previous_word_idx:
                    label_ids.append(index_to_label_dict[predicted_tokens_classes[word_index]])
                    previous_word_idx = word_ids[word_index]
                else:
                    previous_word_idx = word_ids[word_index]
            predicted_labels_for_all_sents.append(label_ids)
    return predicted_labels_for_all_sents

all_labels=predict_labels_for_sentences(model, tokenizer, sents)

#print(len(all_labels), len(sents))

with open("telugu.txt",'w') as f:
     for i in range(len(all_labels)):
         for j in range(len(all_labels[i])):
             f.write(sents[i][j]+"\t"+all_labels[i][j]+"\n")
         f.write("\n")
