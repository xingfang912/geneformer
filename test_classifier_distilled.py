# This file is a distilled version of code to finetune the model
# for cell (disease) classification.
# The evaluation is a 1-step evaluation rather than k-fold

# The distilled version of classification is documented through several key steps.

###############################################################
# Step 1: prepare the training data                           #
# This step should be ran for only once to get the data ready.#
###############################################################
from classifier import Classifier

# select the cell types
filter_data_dict={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]}

# training_args and cc = Classifier initialization is for the purpose of doing the data preparations (cc.prepare_data)
training_args = {
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay":0.258828,
    "per_device_train_batch_size": 12,
    "seed": 73,
}
cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "disease", "states": "all"},
                filter_data=filter_data_dict,
                training_args=training_args,
                max_ncells=None,
                freeze_layers = 2,
                num_crossval_splits = 1,
                forward_batch_size=200,
                nproc=1)


# previously balanced splits with prepare_data and validate functions
# argument attr_to_split set to "individual" and attr_to_balance set to ["disease","lvef","age","sex","length"]
train_ids = ["1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561", "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617", "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371", "1549", "1515"]
eval_ids = ["1422", "1510", "1539", "1606", "1702"]
test_ids = ["1437", "1516", "1602", "1685", "1718"]

train_test_id_split_dict = {"attr_key": "individual",
                            "train": train_ids+eval_ids,
                            "test": test_ids}

output_prefix = "test"
output_dir = "./preprocessed_dataset"
# # Example input_data_file: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
# cc.prepare_data(input_data_file="./human_dcm_hcm_nf.dataset",
#                 output_directory=output_dir,
#                 output_prefix=output_prefix,
#                 split_id_dict=train_test_id_split_dict)


###########################################                       
# Step 2: get the data and model ready
###########################################

import perturber_utils as pu

# 1) Get the data ready: 

train_valid_id_split_dict = {"attr_key": "individual",
                            "train": train_ids,
                            "eval": eval_ids}

prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset"                            

##### Load data and prepare output directory #####
# load numerical id to class dictionary (id:class)
import pickle
with open(f"{output_dir}/{output_prefix}_id_class_dict.pkl", "rb") as f:
    id_class_dict = pickle.load(f)
class_id_dict = {v: k for k, v in id_class_dict.items()}

# load previously filtered and prepared data
nproc = 1 # can be more than 1, up to the number of processers of the cpu
data = pu.load_and_filter(None, nproc, prepared_input_data_file)
data = data.shuffle(seed=42)  # reshuffle in case users provide unshuffled data

# define output directory path
import datetime, subprocess
classifier = "cell"

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
if output_dir[-1:] != "/":  # add slash for dir if not present
    output_dir = output_dir + "/"
output_dir_ = f"{output_dir}{datestamp}_geneformer_{classifier}Classifier_{output_prefix}/"
subprocess.call(f"mkdir {output_dir_}", shell=True)

# get number of classes for classifier
num_classes = len(id_class_dict)

# get the train/eval split ready
import numpy as np

results = []
all_conf_mat = np.zeros((num_classes, num_classes))

data_dict = dict()
data_dict["train"] = pu.filter_by_dict(
    data,
    {train_valid_id_split_dict["attr_key"]: train_valid_id_split_dict["train"]},
    nproc,
)
data_dict["test"] = pu.filter_by_dict(
    data,
    {train_valid_id_split_dict["attr_key"]: train_valid_id_split_dict["eval"]},
    nproc,
)

train_data = data_dict["train"]
eval_data = data_dict["test"]


# further clean up the data, only keeping relevant columns
# this reduces # of columns from 8 to 3
import classifier_utils as cu

# debugging:
# print("8 columns before remove:")
# print(train_data.features.keys())

train_data, eval_data = cu.validate_and_clean_cols(
            train_data, eval_data, classifier)

# debugging:
# print("3 columns after remove:")
# print(train_data.features.keys())           
 

# output a data summary on training
label_set = set(train_data["label"])
print(f"Training data has {len(train_data)} rows,\
with labels being {label_set}, indicating {len(id_class_dict)} diseases: {id_class_dict}")  

# # output a data summary on evaluation\
label_set = set(eval_data["label"])
print(f"Evaluation data has {len(eval_data)} rows, \
with labels being {label_set}, indicating {len(id_class_dict)} diseases: {id_class_dict}")  


# """
# Important:
# 1. I will use the information in train_data["input_ids"] to train a new tokenizer
# """
# tokenized_dataset = train_data["input_ids"]

# # start the training
# final_vocab_size = 40000
# import os
# tokenizer_path = os.getcwd()+"/saved_tokenizers"
# tokenizer_list = os.listdir(tokenizer_path)
# if tokenizer_list == []:
#     print("Train from the beginning...")
#     from Tokenization import GeneTokenizer
#     tokenizer = GeneTokenizer()
#     tokenizer.train(tokenized_dataset,final_vocab_size)
# else:
#     import re
#     tokenizer_list.sort(key=lambda f: int(re.sub('\D','',f)))
#     most_current_tokenizer = tokenizer_list[-1]
#     print(f"Load {most_current_tokenizer}...")
#     import pickle
#     tokenizer = pickle.load(open(tokenizer_path+"/"+most_current_tokenizer,"rb"))
#     tokenizer.train(tokenized_dataset,final_vocab_size)    







"""
2. This is where I can load up my trained tokenizer to change the tokens 
in train_data["input_ids"] and eval_data["input_ids"]
"""

#TBD
#TBD
#TBD



# # 2) get the model ready
# import subprocess, os
# # logger = logging.getLogger(__name__)

# iteration_num = 1 # if 1 then no k-fold cross validation
# ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")

#  # ensure not overwriting previously saved model
# saved_model_test = os.path.join(ksplit_output_dir, "pytorch_model.bin")
# # if os.path.isfile(saved_model_test) is True:
# #     logger.error("Model already saved to this designated output directory.")
# #     raise

# # make output directory
# subprocess.call(f"mkdir {ksplit_output_dir}", shell=True)

# ##### Load model and training args #####
# model_type = "CellClassifier"
# model_directory=r"D:\work\Spring 2024\geneformer\Geneformer\fine_tuned_models\geneformer-6L-30M_CellClassifier_cardiomyopathies_220224"
# model = pu.load_model(model_type, num_classes, model_directory, "train")   

# classifier_type = "cell"
# def_training_args, def_freeze_layers = cu.get_default_train_args(
#             model, classifier, train_data, ksplit_output_dir)

# if training_args is not None:
#     def_training_args.update(training_args)
#     logging_steps = round(
#         len(train_data) / def_training_args["per_device_train_batch_size"] / 10
#     )

# def_training_args["logging_steps"] = logging_steps
# def_training_args["output_dir"] = ksplit_output_dir

# from transformers.training_args import TrainingArguments
# training_args_init = TrainingArguments(**def_training_args)



# def_freeze_layers = 2
# if def_freeze_layers > 0:
#     modules_to_freeze = model.bert.encoder.layer[:def_freeze_layers]
#     for module in modules_to_freeze:
#         for param in module.parameters():
#             param.requires_grad = False

"""
3. Here is where I can change the Embedding and Positional Encoding layers
"""
# import torch
# new_vocab_size = 40000
# d_dimension = 256 # according to the model
# model.bert.embeddings.word_embeddings.weight.data = \
#     torch.tensor(np.random.randn(new_vocab_size,d_dimension),
#     dtype=torch.float32,
#     device="cuda")



###########################################                       
# Step 3: Finetune the model
# The trainer will call the data_collator, which is a DataCollatorForCellClassification instance,
# to handle the padding of data. By default, data is right-padded to the max length of each batch.
###########################################

# from collator_for_classification import DataCollatorForCellClassification
# from transformers import Trainer

# data_collator = DataCollatorForCellClassification()
# # create the trainer
# trainer = Trainer(
#     model=model,
#     args=training_args_init,
#     data_collator=data_collator,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     compute_metrics=cu.compute_metrics,
# )

# trainer.train()
# trainer.save_model(ksplit_output_dir)

###Evaluation Metrics###
labels = id_class_dict.keys()
import evaluation_utils as eu
forward_batch_size = 12

y_pred, y_true, logits_list = eu.classifier_predict(trainer.model, classifier, eval_data, forward_batch_size) 
conf_mat, macro_f1, acc, roc_metrics = eu.get_metrics(y_pred, y_true, logits_list, num_classes, labels)
pred_dict = {"pred_ids": y_pred, "label_ids": y_true, "predictions": logits_list}
from pathlib import Path
pred_dict_output_path = (
    Path(ksplit_output_dir) / f"{output_prefix}_pred_dict"
).with_suffix(".pkl")
with open(pred_dict_output_path, "wb") as f:
    pickle.dump(pred_dict, f)

result = {"conf_mat": conf_mat, "macro_f1": macro_f1, "acc": acc,"roc_metrics": roc_metrics}    
results += result
all_conf_mat = all_conf_mat + result["conf_mat"]
iteration_num = iteration_num + 1

import pandas as pd
all_conf_mat_df = pd.DataFrame(all_conf_mat, columns=id_class_dict.values(), index=id_class_dict.values())
all_metrics = {
            "conf_matrix": all_conf_mat_df,
            "macro_f1": [result["macro_f1"] for result in results],
            "acc": [result["acc"] for result in results]}
all_metrics["all_roc_metrics"] = None

# save evaluation metric
eval_metrics_output_path = (
                Path(ksplit_output_dir) / f"{output_prefix}_eval_metrics_dict"
).with_suffix(".pkl")
with open(eval_metrics_output_path, "wb") as f:
    pickle.dump(all_metrics, f)           



###Evaluate a saved model###

cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "disease", "states": "all"},
                forward_batch_size=200,
                nproc=1)    

all_metrics_test = cc.evaluate_saved_model(
        model_directory=f"./preprocessed_dataset/240408_geneformer_cellClassifier_test/ksplit1/",
        id_class_dict_file=f"./preprocessed_dataset/test_id_class_dict.pkl",
        test_data_file=f"./preprocessed_dataset/test_labeled_test.dataset",
        output_directory="./preprocessed_dataset",
        output_prefix="test",
)

cc.plot_conf_mat(
        conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
        output_directory=".preprocessed_dataset",
        output_prefix="test",
        custom_class_order=["nf","hcm","dcm"],
)

# # 6 layer Geneformer: https://huggingface.co/ctheodoris/Geneformer/blob/main/model.safetensors
# all_metrics = cc.validate(model_directory=r"D:\work\Spring 2024\geneformer\Geneformer\fine_tuned_models\geneformer-6L-30M_CellClassifier_cardiomyopathies_220224",
#                           prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
#                           id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
#                           output_directory=output_dir,
#                           output_prefix=output_prefix,
#                           split_id_dict=train_valid_id_split_dict)
#                           # to optimize hyperparameters, set n_hyperopt_trials=100 (or alternative desired # of trials)                         