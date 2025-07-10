# Topological and Semantic Fusion for Few-Shot Knowledge Graph Completion

### dataset

We adopt Nell and Wiki datasets to evaluate our model, TSF.
The orginal datasets and pretrain embeddings are provided from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). 
For convenience, the datasets can be downloaded from [Nell data](https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz)
and [Wiki data](https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz). 
The pre-trained embeddings can be downloaded from [Nell embeddings](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
 and [Wiki embeddings](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing).
Note that all these files were provided by xiong and we just select what we need here. 
All the dataset files and the pre-trained TransE embeddings should be put into the directory ./data/NELL and ./data/Wiki, respectively.

### How to run

#### NELL-One

```bash
# NELL-One, 5-shot,
python main.py --fine_tune --lr 8e-5 --few 5 --prefix nelllr8e-5.5shot```
```

```bash
# NELL-One, 3-shot,
python main.py --fine_tune --lr 8e-5 --few 3 --prefix nelllr8e-5.3shot```
```

#### Wiki-One

```bash
# Wiki-One, 5-shot,
python main.py --fine_tune --lr 2e-4 --few 5 --prefix wikilr2e-4.5shot```
```

```bash
# Wiki-One, 3-shot,
python main.py --fine_tune --lr 2e-4 --few 3 --prefix wikilr2e-4.3shot```
```
