# TSSDS
The source code of Combine Triplet-Specific Subgraph Modeling and Dual Semantic Modeling for Inductive Relation Prediction


# Requirements
The required packages are listed in `requirement.txt`


# Training

For example, to train the model TSSDS on WN18RR_v1, run the following command:
``` python
python train.py -d WN18RR_v1 -e WN18RR_v1
```
To test TSSDS, run the following commands:
# Hits@10
``` python
python test_rank.py -d WN18RR_v1_ind -e WN18RR_v1
```
# AUC_PR
``` python
python test_auc.py -d WN18RR_v1_ind -e WN18RR_v1
```


# Acknowledgement
Our code refer to the code of [DEKG-ILP](https://github.com/Ninecl/DEKG-ILP). Thanks for their contributions very much.
