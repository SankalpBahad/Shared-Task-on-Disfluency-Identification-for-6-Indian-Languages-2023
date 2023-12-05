from datasets import load_dataset
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, AdamW, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

chosen="xlm-roberta-base"

metric=load_metric("seqeval")

label_list=["B-Alteration","B-edit_R","B-false_R","B-filler_R","B-pet_R","B-repair_R","B-repeat_R","I-Alteration","I-edit_R","I-false_R","I-filler_R","I-pet_R","I-repair_R","I-repeat_R","O"]

id_to_label = {i : label_list[i] for i in range(len(label_list))}
label_to_id = {label_list[i] : i for i in range(len(label_list))}

num_labels = len(label_list)
model = AutoModelForTokenClassification.from_pretrained(chosen, num_labels=len(label_list))

model.to(device)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)

from datasets import DatasetDict, Dataset

def convert_conll_to_dataset(file_path):
    sentences = []
    labels = []
    ids=[]

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sentence = []
        label = []
        id_=0

        for line in lines:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    ids.append(id_)
                    sentence = []
                    label = []
                    id_+=1
            else:
                parts = line.split('\t')
                token = parts[0]
                tag = parts[-1]
                sentence.append(token)
                if tag in label_list:
                    label.append(label_to_id[tag])
                else:
                    label.append(0)

    if sentence:
        sentences.append(sentence)
        labels.append(label)
        ids.append(id_)
#     print(ids[0],ids[-1],len(sentences))
    data = {"id":ids,"tokens": sentences, "ner_tags": labels}
    dataset = Dataset.from_dict(data)

    return dataset


def tokenize_adjust_labels(all_samples_per_split):
  tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], padding='max_length', max_length=128, is_split_into_words=True, truncation=True)

  total_adjusted_labels = []
  
#  for i in tokenized_samples["input_ids"]:
#    print(len(i))

  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = all_samples_per_split["ner_tags"][k]
    i = -1
    adjusted_label_ids = []

    for word_idx in word_ids_list:
      # Special tokens have a word id that is None. We set the label to -100 so they are automatically
      # ignored in the loss function.
      if(word_idx is None):
        adjusted_label_ids.append(-100)
      elif(word_idx!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = word_idx
      else:
        label_name = label_list[existing_label_ids[i]]
        adjusted_label_ids.append(existing_label_ids[i])

    total_adjusted_labels.append(adjusted_label_ids)

  #add adjusted labels to the tokenized samples
  tokenized_samples["labels"] = total_adjusted_labels
  return tokenized_samples

def compute_metrics(p):
    predictions, labels = p

    #select predicted index with maximum logit for each token
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    #languages=['ta']

    #raw_datasets = {}

    #for lang in languages:
    #raw_datasets["ta"] = load_dataset('ai4bharat/naamapadam', "ta")


    #label_list = ['O','B-NEP','I-NEP','B-NEL','I-NEL','B-NEO','I-NEO']
    #label_list=['B-NEL','B-NEO','B-NEP','I-NEL','I-NEO','I-NEP','O']
    #label_list = ['O','B-NEP','I-NEP','B-NEO','I-NEO','B-NEL','I-NEL','B-NETI','I-NETI','B-NEMI','I-NEMI','B-NEN','I-NEN','B-NEAR','I-NEAR','B-NEDA','I-NEDA','B-NETE','I-NETE']
    #label_list = ['B-NEL','B-NEO','B-NEP','I-NEL','I-NEO','I-NEP','O','B-NETI','I-NETI','B-NEMI','I-NEMI','B-NEN','I-NEN','B-NEAR','I-NEAR','B-NEA','I-NEA','B-NEDA','I-NEDA','B-NETE','I-NETE']
    #label_list = ['B-NEL','B-NEO','B-NEP','I-NEL','I-NEO','I-NEP','O','B-NETI','I-NETI','B-NEN','I-NEN','B-NEAR','I-NEAR','B-NEU','I-NEU']
    #label_list = ['O','O','O','B-NEAR','B-NEL','O','B-NEN','B-NEO','B-NEP','O','B-NETI','O','O','O','I-NEAR','I-NEL','O','I-NEN','I-NEO','I-NEP','O','I-NETI','O']
    #label_list=["Alteration","edit_R","false_R","filler_R","pet_R","repair_R","repeat_R","O"]
    label_list=["B-Alteration","B-edit_R","B-false_R","B-filler_R","B-pet_R","B-repair_R","B-repeat_R","I-Alteration","I-edit_R","I-false_R","I-filler_R","I-pet_R","I-repair_R","I-repeat_R","O"]
    id_to_label = {i : label_list[i] for i in range(len(label_list))}
    label_to_id = {label_list[i] : i for i in range(len(label_list))}
    print(label_to_id)

    num_labels = len(label_list)

    #pre_concatenated_train_split = []

    #for lang in raw_datasets:
    #  pre_concatenated_train_split.append( raw_datasets[lang]['train'] )

    #pre_concatenated_validation_split = []

    #for lang in raw_datasets:
    #  pre_concatenated_validation_split.append( raw_datasets[lang]['validation'] )

    from datasets import concatenate_datasets, DatasetDict

    #concatenated_dataset = DatasetDict()
    #concatenated_dataset["train"] = concatenate_datasets(
    #    pre_concatenated_train_split
    #)
    #concatenated_dataset["validation"] = concatenate_datasets(
    #    pre_concatenated_validation_split
    #)
    
    # Replace Language by specific language

    file_1="Language/language_dev.tsv"
    dataset_test=convert_conll_to_dataset(file_1)
    file_2="Language/comb_language_train.tsv"
    dataset_train=convert_conll_to_dataset(file_2)
    concatenated_dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_test,
    })
    #concatenated_dataset = load_dataset("cfilt/HiNER-collapsed")
    #print(concatenated_dataset)
    #exit()
    train_dataset = concatenated_dataset["train"]
    train_dataset = train_dataset.map(
        tokenize_adjust_labels,
        batched=True,
        num_proc=32,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    test_dataset = concatenated_dataset["validation"]
    test_dataset = test_dataset.map(
        tokenize_adjust_labels,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Running tokenizer on test dataset",
    )

    #exit()
    data_collator = DataCollatorForTokenClassification(tokenizer)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    metric = load_metric("seqeval")

    batch_size = 16

    logging_steps = len(concatenated_dataset['train'])

    epochs = 40
    print("training args to be executed")
    training_args = TrainingArguments(
        output_dir="/scratch/sankalp/out_dir",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps)
    print("training args defined")

    # # tokenized_dataset["train"][0]
    # training_dataset=[]
    # for i in tokenized_dataset["train"]:
    #   training_dataset.append(i)
    #   if(len(training_dataset)==10000):
    #     break
    # # training_dataset

    #testing_dataset=[]
    #for i in train_dataset:
    #    testing_dataset.append(i)
    #    if(len(testing_dataset)==100):
    #        break
    # testing_dataset

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("training started")
    trainer.train()

    trainer.evaluate()

    trainer.save_model("Language_model")

if __name__ == '__main__':
    main()
