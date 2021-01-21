# Pairwise Preference Learning 
This repository is based on [DLDH experiments](https://github.com/potamides/dldh-experiments)

## Setup

Clone the repository:
```sh
git clone --recurse-submodules https://github.com/ndarr/pairwise-preference-learning.git
```

Install requirements:

```sh
pip install -r requirements.txt
```

The annotated pairwise dataset should belong in data_poems and should have the following format:
```
id, poem1, poem2, dataset1, dataset2, category1, category2, ..., categoryN
```
The number of categories can be arbitrary as they are seen as multiple entries with preferences when trained on multiple categories.
If it is desired to take only a subset of poems into account put the selection into *data_poems/multi_annotated_poems_10.txt*. Only pairs in the csv containing at least once one of the poems in this subset are considered for training.

## Training
Models can be trained by executing their respective python scripts.
Using the parameter *--subset* allows the training only on pairs which contain poems selected with the *subset.ipynb* code.<br>
*--no-training* skips the training process and does not update the score files. It just produces the accuracy output of the earlier trained model. This option is only available if training has been done earlier or the respective model files are in the model directory. 
### BWS
```shell
python bws.py [--subset]
```

### GPPL
```shell
python gppl.py [--subset] [--no-training]
```

### crowdGPPL
```shell
python crowdgppl.py [--subset] [--no-training]
```

### BERTRanker
```shell
python bertranker.py [--subset] [--no-training]
```

## Results
Results can be examined by taking a look at the various score files. These files are located in the directory *scores* and can be merged into one file using the code in *merge_method_scores.ipynb*. The file *scores/normalized_scores.csv* only contains the poems from the subset as they led to better results in downstream applications<br>
Further the accuracy of the methods on how good they replicate the pairwise preferences of the annotators can be examined by using *accuracy.ipynb*.<br>
For experiments on only a subset of pairs a file containing the poems which have been annotated 10 times or more can be collected by executing *subset.ipynb*. The generated files are then considered in the code if the respective flags are set.
