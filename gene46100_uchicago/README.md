# In-Silico Perturbation Studies using Genformer for Understanding Cellular Response to Radiation

This repository contains code for performing **in-silico perturbation studies** to understand the genes responsible for shifting transcriptional states in response to external radiation exposure.

## Pipeline Overview

1. **Download Single-Cell Datasets**  
   Obtain single-cell RNA-seq dataset from [GSE255800](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255800) relevant to radiation exposure studies.

2. **Quality Control & Preprocessing**  
   Perform quality control and save the dataset as an `.h5ad` file using the following Google Colab notebook:  
   [QC & Save Notebook](https://colab.research.google.com/drive/1DnmUwohTgF5E9LZjQR5QfhyNk19IAWdQ?usp=sharing)

3. **Tokenize the Data**  
   Use `tokenize_data.py` to convert the `.h5ad` file into a tokenized format suitable for input into the Geneformer model.

4. **Fine-Tune Geneformer**  
   Fine-tune the pretrained Geneformer model to distinguish between cells exposed to different radiation levels.

5. **Visualize Cell Embeddings**  
   Use `extract_and_plot_cell_embeddings.py` to extract and visualize the learned cell embeddings.

6. **In-Silico Perturbation Studies**  
   Run `isp.py` to perform in-silico perturbations and identify genes driving transcriptional changes.

---

For questions or issues, please contact tnnandi@gmail.com
