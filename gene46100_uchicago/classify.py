import datetime
from geneformer import Classifier
import numpy as np
from datasets import load_from_disk
import os
from pdb import set_trace
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "radiation"
output_dir = f"/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago/classification_output/{datestamp}"
# output_dir = f"/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago/classification_output/debug"
os.makedirs(output_dir, exist_ok=True)

tokenized_dataset = "/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago/filtered_dataset/68k_cells/tokenized_68k_cells_new.dataset"

training_args = {
    "num_train_epochs": 2.0,
    "learning_rate": 0.0005,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 10,
    "weight_decay":0.25,
    "per_device_train_batch_size": 24,
    "seed": 73,
    # "save_strategy": "epoch",
    "save_safetensors": True,

}

cc = Classifier(classifier="cell",
                cell_state_dict = {"state_key": "radiation_level", "states": ["control", "r100"]},
                filter_data={"radiation_level": ["control", "r100"]},
                training_args=training_args,
                max_ncells=10000,
                freeze_layers = 11,
                num_crossval_splits = 1,
                forward_batch_size=20,
                nproc=16)

# creates the train and test datasets
cc.prepare_data(input_data_file=tokenized_dataset,
                output_directory=output_dir,
                output_prefix=output_prefix,
                test_size=0.2,)

ds = load_from_disk(tokenized_dataset)
print(f"Dataset size: {len(ds)}")
max_token_id = max(max(seq) for seq in ds["input_ids"])
print(f"Max token ID: {max_token_id}") 

# check the model
model_path = "/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gf-12L-95M-i4096"
model = AutoModel.from_pretrained(model_path)
print("model_config: ", model.config)
# set_trace()

# all_metrics = cc.validate(model_directory="/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gf-12L-95M-i4096",
#                           prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
#                           id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
#                           output_directory=output_dir,
#                           output_prefix=output_prefix,)

# set_trace()

print("======================  Fine tuning model  ======================")
# try except added to catch errors related to not returning tokenizer
try:
    all_metrics = cc.validate(
        model_directory="/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gf-12L-95M-i4096",
        prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
except Exception as e:
    print(f"Validation failed: {e}")
    all_metrics = None  # Or use {} or some placeholder

# set_trace()
# edit the checkpoint path to the one you want to evaluate
print("=======================  Evaluating saved model  =======================")
all_metrics_test = cc.evaluate_saved_model(
        model_directory=f"{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/checkpoint-594", # 297 (10000 cells), 593 (20000 cells)
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )

cc.plot_conf_mat(
        conf_mat_dict={"Geneformer": all_metrics_test["conf_matrix"]},
        output_directory=output_dir,
        output_prefix=output_prefix,
        # custom_class_order=["control","r10","r100", "r1000"],
        custom_class_order=["control","r100"],
)

cc.plot_predictions(
    predictions_file=f"{output_dir}/{output_prefix}_pred_dict.pkl",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    title="radiation effects",
    output_directory=output_dir,
    output_prefix=output_prefix,
    # custom_class_order=["control","r10","r100", "r1000"],
    custom_class_order=["control","r100"],
)

print(all_metrics_test)
set_trace()