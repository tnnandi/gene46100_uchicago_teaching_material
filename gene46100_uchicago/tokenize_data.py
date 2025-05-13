from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer({"cell_index":"cell_index", 
                             "radiation_level": "radiation_level"}, 
                             nproc=16)
tk.tokenize_data("/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago/filtered_dataset/68k_cells", # dataset directory
                 "/grand/GeomicVar/tarak/Geneformer_gene46100/Geneformer/gene46100_uchicago/filtered_dataset/68k_cells", # output directory
                 "tokenized_68k_cells_new", # output prefix
                 file_format="h5ad")

