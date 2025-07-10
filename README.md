# Topological and Semantic Fusion for Few-Shot Knowledge Graph Completion
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
