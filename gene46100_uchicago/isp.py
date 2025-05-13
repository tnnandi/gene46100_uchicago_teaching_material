from geneformer import InSilicoPerturber
from geneformer import InSilicoPerturberStats
from geneformer import EmbExtractor
import os
from pdb import set_trace

root_dir = "/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago"

# first obtain start, goal, and alt embedding positions
# this function was changed to be separate from perturb_data
# to avoid repeating calcuations when parallelizing perturb_data
cell_states_to_model={"state_key": "label", 
                      "start_state": [1], 
                      "goal_state": [0], 
                    #   "alt_states": [0, 2]
                      } # 0:r100, 1:r10, 2: r1000, 3: control

filter_data_dict={"label":[0, 1]}

embex = EmbExtractor(model_type="CellClassifier", # if using previously fine-tuned cell classifier model
                     num_classes=2,
                     filter_data=filter_data_dict,
                     max_ncells=1000,
                     emb_layer=0,
                     summary_stat="exact_mean",
                     forward_batch_size=32,
                     nproc=16)

state_embs_dict = embex.get_state_embs(cell_states_to_model,
                                       os.path.join(root_dir, "classification_output/250512075246/250512_geneformer_cellClassifier_radiation/ksplit1/"),
                                       os.path.join(root_dir, "classification_output/250512075246/radiation_labeled_test.dataset"),
                                       os.path.join(root_dir, "classification_output/250512075246"),
                                       "fine_tuned",)


# set_trace()
isp = InSilicoPerturber(perturb_type="delete",
                        perturb_rank_shift=None,
                        genes_to_perturb="all",
                        combos=0,
                        anchor_gene=None,
                        model_type="CellClassifier", # if using previously fine-tuned cell classifier model
                        num_classes=2,
                        # emb_mode="cell",
                        emb_mode = "cls",
                        # emb_label=["label"],
                        cell_emb_style="mean_pool",
                        filter_data=filter_data_dict,
                        cell_states_to_model=cell_states_to_model,
                        state_embs_dict=state_embs_dict,
                        max_ncells=100,
                        emb_layer=0,
                        forward_batch_size=64,
                        nproc=1)

isp.perturb_data(os.path.join(root_dir, "classification_output/250512075246/250512_geneformer_cellClassifier_radiation/ksplit1/"),
                 os.path.join(root_dir, "classification_output/250512075246/radiation_labeled_test.dataset"),
                 os.path.join(root_dir, "classification_output/250512075246/isp"),
                 "fine_tuned")

ispstats = InSilicoPerturberStats(mode="goal_state_shift",
                                  genes_perturbed="all",
                                  combos=0,
                                  anchor_gene=None,
                                  cell_states_to_model=cell_states_to_model)

ispstats.get_stats(os.path.join(root_dir, "classification_output/250512075246/isp"), # this should be the directory 
                   None,
                   os.path.join(root_dir, "classification_output/250512075246/isp_stats"),
                   "fine_tuned",)