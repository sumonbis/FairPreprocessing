# Installation and Usage

The source code is written in Python 3 and tested using Python 3.7 on Mac OS. It is recommended to use a virtual Python environment for this setup. Furthermore, we used bash shell scripts to automate running benchmark and Python scripts.

## Installation and Environment Setup
Follow the instructions below to setup environment and run the source code for reproduction.

Follow these steps to create a virtual environment:

1. Install Anaconda [[installation guide](https://docs.anaconda.com/anaconda/install/)].

2. Create a new Python environment. Run on command line:
```
conda create --name fairpreprocess python=3.7
conda activate fairpreprocess
```
The shell should look like: `(fairpreprocess) $ `. Now, continue to step 2 and install packages using this shell.
When you are done with the experiments, you can exit this virtual environment by `conda deactivate`.

3. Clone this FairPreprocessing repository. This will clone both data and code to run the benchmark.
```
git clone https://github.com/sumonbis/FairPreprocessing.git
```

4. Navigate to the cloned repository: `cd FairPreprocessing/` and install required packages:
```
pip install -r requirements.txt
```


## Run Experiments

#### Run benchmark pipelines
Navigate to the benchmark directory `cd benchmark/`.
Under each of the 5 tasks (`german`, `adult`, `bank`, `compas`, `titanic`), there are separate Python scripts for each pipeline. Running the pipelines will output accuracy and F1 score in command line.

* To run any of the pipelines, run the shell script `./pipeline.sh <task> <pipeline-id>`. For example, to run the pipeline `GC1`, run `./pipeline.sh german GC1`. The performance of the pipelines will be printed in the terminal.

* To run all the pipelines in a task, run `./pipeline.sh <task>`. Depending on the size of the dataset, it might take some time. For testing, `german` can be used as a task since the dataset size is small.

* To run all the pipelines, run `./pipeline.sh all`. However, it might take many hours to finish.  

#### Run analysis (fair preprocessing stages)
Navigate to the source code directory `FairPreprocessing/src/fair-preprocessing/`. Then run shell script to run pipelines *n* times.

* To run a single pipeline: `./stages.sh <task> <pipeline-id> <positive-integer>`. We run these experiments multiple times and accumulate results. For quick testing, run `./stages.sh german GC1 1`.

* To run all the pipelines in a task use this command: `./stages.sh <task> <positive-integer>`.

Running each model will produce result into `.csv` file in this location: `FairPreprocessing/src/fair-preprocessing/<task>/res/`.
The results are then accumulated to `rq1-x.csv` in `FairPreprocessing/result/`.

#### Run analysis (fair transformers)
Navigate to the source code directory `FairPreprocessing/src/fair-transformers/`. Run the shell script to run each task multiple times: `./trans.sh <task> <positive-number>`.

This will automatically run for 5 different classifiers and save results in `FairPreprocessing/src/fair-transformers/res`. The results are accumulated into the spreadsheets of `rq2-x` in `FairPreprocessing/result/`.

## Results

The results are then accumulated into the `.csv` files for analysis. All the results presented in the paper can be found [here](res/). RQ1 and RQ2 directly comes from the results generated by the experiments. RQ3 results come from the subset of the generated results.
