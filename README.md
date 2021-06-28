This repository houses all code for reproducing the results of the paper:

```
CLIP: A Dataset for Extracting Action Items for Physicians from Hospital Discharge Notes
James Mullenbach, Yada Pruksachatkun, Sean Adler, Jennifer Seale, Jordan Swartz, Greg McKelvey, Hui Dai, Yi Yang and David Sontag
ACL 2021
```

# Setup

1. (Optional, recommended) Create and activate a virtual environment
2. `pip install -r requirements.txt`. 

# Data

[Link](https://physionet.org/content/mimic-iii-clinical-action/1.0.0/)

The data is hosted by PhysioNet, as it is based on MIMIC-III, which has access controls for privacy reasons and which is also hosted by PhysioNet. Interested users will have to complete a short training course and sign a DUA with PhysioNet to gain access.

# Data processing

After downloading and unpacking the data, `cd data` and run `python convert_json_to_csv.py` to create sentence-level data that the training scripts can read.

[step to split into train/val/test]

# Experiments

All training scripts are in `sentclf/`. Bash scripts to invoke these scripts in the manner they were run to produce the paper results are found in `train_{modelname}.sh`. With the exception of the bag of words model, these scripts all will run training for multiple seeds and report final results on the test set, which can then be used to compute the table entries from the paper. 

# Pre-training

First, sentence-tokenize each note in MIMIC's `NOTEEVENTS.csv` file. Then, run `apply_bert_unlabeled.py` with a fine-tuned vanilla BERT model (trained with `sentclf/bert.py`) and the tokenized unlabeled data as inputs. This creates a data file with three columns: `document_id, sentence_index, max_label_score`. One can then use this file, sorted by decreasing `max_label_score`, to create task-targeted pre-training datasets of a given size. For example, taking the top 250k sentences by `max_label_score` will give (approximately) the same dataset used to pre-train the TTP-BERT+Context (250k) model from the paper. 

Once you have this task-targeted dataset, use `sentclf/pretrain_tasks.py` to pretrain a model on the dataset. Example invocation for 250k sentences:
```
python pretrain_tasks.py
--fname ../data/unlabeled_tokenized_sentences.csv
--model bert
--focus_fname ../data/ttp_doc_sents_250k.csv
--task mlm_switch
--max_epochs 100
--mlm_probability 0.15
--n_context_sentences 2
--switch_prob 0.25
--patience 5
--lr 5e-5
--batch_size 8
--seed 11
--print_every 100
--gradient_accumulation_steps 4
--eval_steps 3000
```

Finally, one can then replace the pointers to `local_weights` in the `train_pretrained_{size}.sh` scripts to point to the newly TTP pre-trained models and roughly reproduce paper results (there will likely be some non-determinism with the final results here). 
