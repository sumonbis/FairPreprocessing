# Fair Preprocessing
This repository contains the source code and data used for the [paper](/fair-preprocessing-paper.pdf), to be appeared at ESEC/FSE 2021. For any questions, contact the corresponding author. The replication package is licensed under [MIT License](/LICENSE.md): Copyright (c) 2020 Sumon Biswas

**Title** Fair Preprocessing: Towards Understanding Compositional Fairness of Data Transformers in Machine Learning Pipeline

**Authors** Sumon Biswas (sumon@iastate.edu) and Hridesh Rajan (hridesh@iastate.edu)

## Index
> 1. [Benchmark](#benchmark)
> 2. [Installation and Evaluation](/INSTALL.md)
> 3. Datasets
  >> * [German Credit](data/german)
  >> * [Adult Census](data/adult)
  >> * [Bank Marketing](data/bank)
  >> * [Compas](data/compas)
  >> * [Titanic](data/titanic)
> 4. Source code
  >> * Experiments
  >>> + [Fair preprocessing](src/fair-preprocessing)
  >>> + [Fair transformers](src/fair-transformers)
  >> * [Utilities](utils/)
>  5. [Results](res/) (RQ1, RQ2, RQ3)
>  6. [DOI and Citation](#doi-of-replication-package)


## Benchmark
The benchmark contains 37 ML pipelines under 5 different tasks from three prior studies.

| German Credit | Adult Census | Bank Marketing | Compas | Titanic |
|---------------|--------------|----------------|--------|---------|
| [GC1](benchmark/german/GC1.py) | [AC1](benchmark/adult/AC1.py) | [BM1](benchmark/bank/BM1.py) | [CP1](benchmark/compas/CP1.py) | [TT1](benchmark/titanic/TT1.py) |
| [GC2](benchmark/german/GC2.py) | [AC2](benchmark/adult/AC2.py) | [BM2](benchmark/bank/BM2.py) | - | [TT2](benchmark/titanic/TT2.py) |
| [GC3](benchmark/german/GC3.py) | [AC3](benchmark/adult/AC3.py) | [BM3](benchmark/bank/BM3.py) | - | [TT3](benchmark/titanic/TT3.py) |
| [GC4](benchmark/german/GC4.py) | [AC4](benchmark/adult/AC4.py) | [BM4](benchmark/bank/BM4.py) | - | [TT4](benchmark/titanic/TT4.py) |
| [GC5](benchmark/german/GC5.py) | [AC5](benchmark/adult/AC5.py) | [BM5](benchmark/bank/BM5.py) | - | [TT5](benchmark/titanic/TT5.py) |
| [GC6](benchmark/german/GC6.py) | [AC6](benchmark/adult/AC6.py) | [BM6](benchmark/bank/BM6.py) | - | [TT6](benchmark/titanic/TT6.py) |
| [GC7](benchmark/german/GC7.py) | [AC7](benchmark/adult/AC7.py) | [BM7](benchmark/bank/BM7.py) | - | [TT7](benchmark/titanic/TT7.py) |
| [GC8](benchmark/german/GC8.py) | [AC8](benchmark/adult/AC8.py) | [BM8](benchmark/bank/BM8.py) | - | [TT8](benchmark/titanic/TT8.py) |
| [GC9](benchmark/german/GC9.py) | [AC9](benchmark/adult/AC9.py) | - | - | - |
| [GC10](benchmark/german/GC10.py) | [AC10](benchmark/adult/AC10.py) | - | - | - |

## DOI of Replication Package
[![DOI](https://zenodo.org/badge/371777846.svg)](https://zenodo.org/badge/latestdoi/371777846)

Cite the paper as:

```
@inproceedings{biswas21fair,
  author = {Sumon Biswas and Hridesh Rajan},
  title = {Fair Preprocessing: Towards Understanding Compositional Fairness of Data Transformers in Machine Learning Pipeline},
  booktitle = {ESEC/FSE'2021: The 29th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  location = {Athens, Greece},
  year = {2021},
  entrysubtype = {conference},
  url = {https://doi.org/10.1145/3468264.3468536},
}
```
