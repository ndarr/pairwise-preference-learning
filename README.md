# Pairwise Preference Learning 
This repository is based on [DLDH experiments](https://github.com/potamides/dldh-experiments)

## Setup

Clone the repository:
```sh
git clone --recurse-submodules git@github.com:ndarr/pairwise-preference-learning.git
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
Real poems shall be put into *real_poems/real_poems.txt* seperated with one poem in each line (*\<br>* as line break inside poems).

## Training
Models can be trained by executing their respective python scripts and allow a variety of parameters.
Sequence embeddings are saved after the first run and loaded on each additional run. This can be omitted by providing *--fresh* as parameter. The newly created sequence embeddings are written over the old ones.
### BWS
```shell
python bws.py
```

### GPPL
```shell
python gppl.py [--fresh] [--subset]
```

### crowdGPPL
```shell
python crowdgppl.py [--fresh] [--subset]
```

### BERTRanker
```shell
python bertranker.py [--fresh] [--subset]
```

## Results
Results can be examined by taking a look at the various score files. These files are located in the directory *scores* and can be merged into one file using the code in *merge_method_scores.ipynb*. <br>
Further the accuracy of the methods on how good they replicate the pairwise preferences of the annotators can be examined by using *accuracy.ipynb*.<br>
For experiments on only a subset of pairs a file containing the poems which have been annotated 10 times or more can be collected by executing *subset.ipynb*. The generated files are then considered in the code if the respective flags are set.
