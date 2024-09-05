# 2024Chinasoft

This repo provides the code for reproducing the experiments of the paper *An Empirical Study on the Capacity of Code Naturalness Modeling for Large Language Models in Program Repair*.

### Dependency

- pip install torch
- pip install transformers

### Reproduce

#### RQ1

To reproduce the experiment align with n-gram models, please refer:

```shell
python rq1.py --model_name_or_path Salesforce/codet5-base \
--work_mode n-gram \
--max_batch_size 32
```

To reproduce the experiment with custom mask style and context window size, please refer:

```shell
python rq1.py --model_name_or_path Salesforce/codet5-base \
--work_mode custom \
--mask_style MLM \
--max_batch_size 32 \
--context_token_number 100
```

If you wanna further analysis of results, you can redirect output to a .txt file.

#### RQ2

To reproduce the experiment of RQ2, please refer:

```shell
python rq2.py --model_name_or_path Salesforce/codet5-base \
--work_mode partial \
--mask_style MLM \
--max_batch_size 32 \
--context_token_number 100
```

#### RQ3

To reproduce the experiment of RQ3, please refer:

```shell
python rq2.py --model_name_or_path Salesforce/codet5-base \
--mask_style MLM \
--max_batch_size 32 \
--context_token_number 100
```

### Analyze Experiment Results

Code of figures can be found under the directory "plot". 

```shell
cd plot
python RQ1_n_gram.py
```

Before analyzing results of RQs, please reproducing corresponding experiments.

To get mean entropy of common codes, buggy codes or fixed codes, please refer:

```shell
python analyze.py --work_mode statistic \
--result_file [path to your saved .csv file]
```

To get significance measured by Wilcoxon rank sum test, Cohen's d value and our defined SNV value, please refer:

```shell
python analyze.py --work_mode significance \
--result_file [path to your saved .txt file]
```

If our defined CNM value is required, please refer:

```shell
python analyze.py --work_mode significance \
--result_file [path to your saved .txt file] \
--capacity_type CNM
```

Attention that if our defined CDF value is required, two result file paths are needed:

```shell
python analyze.py --work_mode significance \
--result_file [path to your saved .txt file of complete fix result] \
--capacity_type CDF \
--result_file_partial [path to your saved .txt file of partial fix result]
```

To measure the performance of entropy-based APCA technique, please refer:

```shell
python analyze.py --work_mode performance \
--result_file [path to your saved .csv file]
```

