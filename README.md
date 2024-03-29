# WatClaimCheck

This repo contains information about two main sub-projects:

1. Dataset generation - code is under dataset_generation folder
2. Models - code is under models folder

Additionally, it also contains information on how to request access to the dataset.

Upon usage of the WatClaimCheck dataset, please make sure to cite the paper that describes the dataset:

**Khan, K., Wang, R., & Poupart, P. (2022, May). WatClaimCheck: A new Dataset for Claim Entailment and Inference. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1293-1304).**
https://aclanthology.org/2022.acl-long.92.pdf

## Dataset request

The WatClaimCheck dataset is available upon request for non-commercial research purposes only under the Fair Dealing
exception of the Canadian Copyright Act.  Please submit the following form to receive a copy of the dataset: https://forms.gle/sEZjvJqmyHdR4AMKA

Third party materials included in this dataset have been included using the Fair Dealing exception in the Canadian Copyright Act. If you believe your work is included in this dataset and would like us to remove it, please let us know at ppoupart@uwaterloo.ca.

## Dataset generation

### Requirements
- Requests
- BeautifulSoup
- tqdm 
- numpy
- nltk
- scikit learn
- pandas

### Generating dataset

**Data collection** 

For data collection the main script to run is `dataset_generation/data_collection/main.py`. The arguments to the script can be used to control the data source and the type of data retrived. The type of data retrieved can be one ofthe following three types: claim metadata, review article, and relevant articles. The data from each data source should be retrieved in the following order: 1) claim metadata, 2) review article, and finally 3) relevant articles. The config file (`dataset_generation/data_collection/config.conf`) can be updated to set the google api key and to specify the data folder and the data file names.

**Data cleaning** 
1. The config file (`dataset_generation/data_cleaning/config.conf`) contains configuration options specifying raw data folder, dataset folder path, metadata file name, dataset articles folder name, mininum number of articles required for each claim, training set size proportion, etc. Rating mapping file(`dataset_generation/data_cleaning/rating_mappings.py`) contains mapping from refined claim rating to the more broad three class rating (False, Partly True or False, and True).
2. The generate dataset script (`dataset_generation/data_cleaning/generate_dataset.py')
3. The generate dataframes script (`dataset_generation/data_cleaning/generate_dataframes.py')
4. The generate DPR dataset script (`dataset_generation/data_cleaning/generate_dpr_dataset.py')

## Models

### Requirements
- Natsort
- nltk
- numpy
- pandas
- Requests
- scikit learn
- scipy
- pytorch
- tqdm
- transformers

We list below the models whose results were presented in the paper along with the associated script file which can be used to train the model:

1. Roberta-base (pooled) model: `models/Roberta_baseline.py`
2. Roberta-base (averaged) model: `models/Roberta_weighted_baseline.py`
3. Roberta-base (pooled) model using DPR dataframe: `models/Roberta_DPR_baseline.py`
4. Prequential Roberta-base (pooled) using DPR dataframe: `models/Prequential_roberta_dpr_pooled.py`
5. Prequential Roberta-base (averaged) using DPR dataframe: `models/Prequential_roberta_dpr_averaged.py`
6. DPR (training): `models/DPR.py`
7. DPR (inference script for generating dataframe for second stage): `models/DPR_inference.py`



## Hierarchical Attention Networks:
### Requirements
- datetime
- dateutil
- h5py
- keras
- nltk
- numpy
- pandas
- tensorflow
- tqdm

Before running HAN models, use data generation script to generate `train.pkl`, `valid.pkl` and `test.pkl` in the data directory. HAN models can be simply run by:

1. HAN-base model (Bi-LSTM): `models/HAN_baseline.py`
2. HAN-prequential model (Bi-LSTM): `models/Prequential_HAN.py`
