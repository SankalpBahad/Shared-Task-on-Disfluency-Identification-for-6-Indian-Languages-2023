# Shared Task on Disfluency Identification for 6 Indian Languages 2023

## Approach and Data

We finetuned XLM-Roberta-Base model on the dataset shared for this task. We performed 3 experiments varying the number of epochs,
to 10, 20 and 40 epochs.
We only used the dataset released for this shared task.
XLM-Roberta-Base model works well for tasks belonging to this family, like Named Entity Recognition, Part of Speech tagging, etc. 
We used similar approaches to NER and POS. xlm-roberta model has wide representations of all Indian languages and is pretrained on high quality
data, which enabled us to choose it for this task. 
