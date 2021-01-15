# TODO
- Only one submodule for gppl and crowdGPPL
- Integrate conf matrices 
- Make all modules trainable and with a pipeline which makes sense

# GPPL Comparison
This repository is based on [DLDH experiments](https://github.com/potamides/dldh-experiments)

## Setup

Clone the repository:
```sh
git clone --recurse-submodules git@github.com:DrCracket/dldh-experiments.git
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
By default all preferences from all categories are used but training can be also done by selecting a single category with *--cat N* starting with 0.
Sequence embeddings are saved after the first run and loaded on each additional run. This can be omitted by providing *--fresh* as parameter. The newly created sequence embeddings are written over the old ones.
### BWS
```shell
python bws.py [--fresh] [--cat N]
```

### GPPL
```shell
python gppl.py [--fresh] [--cat N]
```

### crowdGPPL
```shell
python crowdgppl.py [--fresh] [--cat N]
```

### BERTRanker
```shell
python bertranker.py [--fresh] [--cat N]
```

## Experiments

Train models and compute features:
```sh
python gppl.py && python crowdgppl.py
```

Run the experiments:
```sh
python main.py
```

Compute various statistics of the dataset:
```sh
python statistics.py
```

An overview of the results can be found [here](https://hackmd.io/TqQqj5r1QoS-MoTSbYpQ5g).
