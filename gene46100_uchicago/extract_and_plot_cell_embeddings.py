from geneformer import EmbExtractor
import os
from pdb import set_trace

root_dir = "/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago"

# initiate EmbExtractor
embex = EmbExtractor(model_type="CellClassifier",
                     num_classes=2,
                    #  filter_data={"label": [1, 3]},
                    #  max_ncells=959,   # [1,3]: 959, [0,3]: 903
                    #  max_ncells=10000,
                     emb_layer=0,
                     emb_label=["label"],
                     labels_to_plot=["label"],
                     forward_batch_size=50,
                     nproc=16,)

embs = embex.extract_embs(
    os.path.join(root_dir, "classification_output/250512075246/250512_geneformer_cellClassifier_radiation/ksplit1/"),  
    # "/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gf-20L-95M-i4096",
    os.path.join(root_dir, "classification_output/250512075246/radiation_labeled_test.dataset"),
    os.path.join(root_dir, "classification_output/250512075246"),
    # "pretrained", 
    "fine_tuned",
)


# set_trace()
embex.plot_embs(
    embs=embs,
    output_directory=os.path.join(root_dir, "classification_output/250512075246"),
    # output_prefix="pretrained", 
    output_prefix="fine_tuned",
    plot_style="umap",
    kwargs_dict={"cmap": "plasma"}
)