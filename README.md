# data-deduplication

Repo for the data deduplication project. 

### Problem Statement

“The presence of duplicates can potentially dilute the signal strength that one can extract from the column for ML” [1] which can affect the accuracy of models trained on the data. To mitigate this loss, we aim to study the statistical features of data and duplicate category points, and build a model for detecting categorical duplicates in a table. 

### Structure

The data-extraction folder contains scripts in jupyter notebook to generate word pairs for our classification problem and the data itself in csv format.

The code folder contains Jupyter Notebooks to Train and Test various approaches used for the binary classification task. The dedup_lib contains the required functions and classes. 

Install the requirements according to the requirements.txt file


### Todo

- [x] Tests on n-gram and Distance Features
- [x] Cross Validation
- [ ] Function to call the pretrained classifier to clean up Pandas DataFrame


### Relevent Papers

[1] An Empirical Study on the (Non-)Importance of Cleaning Categorical Duplicates before ML \
Vraj Shah, Thomas Parashos, and Arun Kumar

[2] Towards Benchmarking Feature Type Inference for AutoML Platforms \
Vraj Shah, Kevin Yang, and Arun Kumar