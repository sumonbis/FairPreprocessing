# Fair Preprocessing
This repository contains the benchmark pipelines, source code and data for the paper on "Fair Preprocessing".

### Index

1. [Benchmark](#benchmark)
2. [Installation and Environment Setup](#installation-and-environment-setup)
3. [Run Experiment](#run-experiment)
4. Datasets
  - [German Credit](data/german)
  - [Adult Census](data/adult)
  - [Bank Marketing](data/bank)
  - [Compas](data/compas)
  - [Titanic](data/titanic)
5. Source code
  - Experiments
    - [Fair preprocessing](src/fair-preprocessing)
    - [Fair transformers](src/fair-transformers)
  - [Utilities](utils/)
6. [Results](res/) (RQ1, RQ2, RQ3)


### Benchmark
The benchmark contains 37 ML pipelines under 5 different tasks from three prior studies.

1. **German Credit (GC)**
  - [GC1](benchmark/german/GC1.py)
  - [GC2](benchmark/german/GC2.py)
  - [GC3](benchmark/german/GC3.py)
  - [GC4](benchmark/german/GC4.py)
  - [GC5](benchmark/german/GC5.py)
  - [GC6](benchmark/german/GC6.py)
  - [GC7](benchmark/german/GC7.py)
  - [GC8](benchmark/german/GC8.py)
  - [GC9](benchmark/german/GC9.py)
  - [GC10](benchmark/german/GC10.py)
2. **Adult Census (AC)**
  - [AC1](benchmark/adult/AC1.py)
  - [AC2](benchmark/adult/AC2.py)
  - [AC3](benchmark/adult/AC3.py)
  - [AC4](benchmark/adult/AC4.py)
  - [AC5](benchmark/adult/AC5.py)
  - [AC6](benchmark/adult/AC6.py)
  - [AC7](benchmark/adult/AC7.py)
  - [AC8](benchmark/adult/AC8.py)
  - [AC9](benchmark/adult/AC9.py)
  - [AC10](benchmark/adult/AC10.py)
3. **Bank Marketing (BM)**
  - [BM1](benchmark/bank/BM1.py)
  - [BM2](benchmark/bank/BM2.py)
  - [BM3](benchmark/bank/BM3.py)
  - [BM4](benchmark/bank/BM4.py)
  - [BM5](benchmark/bank/BM5.py)
  - [BM6](benchmark/bank/BM6.py)
  - [BM7](benchmark/bank/BM7.py)
  - [BM8](benchmark/bank/BM8.py)
4. **Compas (CP)**
  - [CP1](benchmark/compas/CP1.py)
5. **Titanic (TT)**
  - [TT1](benchmark/titanic/TT1.py)
  - [TT2](benchmark/titanic/TT2.py)
  - [TT3](benchmark/titanic/TT3.py)
  - [TT4](benchmark/titanic/TT4.py)
  - [TT5](benchmark/titanic/TT5.py)
  - [TT6](benchmark/titanic/TT6.py)
  - [TT7](benchmark/titanic/TT7.py)
  - [TT8](benchmark/titanic/TT8.py)


### Installation and Environment Setup
Follow the instructions below to setup environment and run the source code for reproduction.

The source code is written in Python 3 and tested using Python 3.7.
It is recommended to use a virtual Python environment for this setup.

Follow the steps to create a virtual environment and run the code.

1. Install Anaconda [[installation guide](https://docs.anaconda.com/anaconda/install/)].

2. Create a new Python environment. Run on command line:
```
conda create --name fairpreprocess python=3.7
conda activate fairpreprocess
```
The shell should look like: `(fairpreprocess) $ `. Now, continue to step 2 and install packages using this shell.
When you are done with the experiments, you can exit this virtual environment by `conda deactivate`.

3. Clone this FairPreprocessing repository.
```
git clone https://github.com/anonymous-authorss/FairPreprocessing.git
```
It will clone both data and code to run the benchmark.

4. Install required packages:
```
pip install xgboost imblearn aif360 catboost lightgbm
```

### Run Experiment

#### Run benchmark pipelines
Navigate to the benchmark directory `FairPreprocessing/benchmark/`.

Under each of the 5 tasks (`german`, `adult`, `bank`, `compas`, `titanic`), there are separate Python scripts for each pipeline.
To run any of the pipelines, run `python <pipeline-id>.py`. For example, to run the pipeline `GC1`, run `python GC1.py`.

#### Run analysis (fair preprocessing stages)
Navigate to the source code directory `FairPreprocessing/src/fair-preprocessing/`. Then, for each pipeline, there is separate Jupyter Notebook, e.g., `GC1.ipynb`. Run all the cells of the notebook.
Running each model will produce result into `.csv` file in this location: `FairPreprocessing/src/fair-preprocessing/res`.
Experiments are conducted 10 times and then accumulated.

#### Run analysis (fair transformers)
Navigate to the source code directory `FairPreprocessing/src/fair-transformers/`. For each dataset, there is a separate Jupyter Notebook, e.g., `compas-transformers.ipynb`. Run all the cells of each notebook.
Running each model will produce result into `.csv` file in the corresponding directory. These experiments are also conducted multiple times.

The results are then accumulated into the `.csv` files for analysis. All the results presented in the paper can be found in [here](res/).
