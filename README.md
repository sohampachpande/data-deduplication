# data-deduplication

Repo for the data deduplication project. 

### Problem Statement

“The presence of duplicates can potentially dilute the signal strength that one can extract from the column for ML” [1] which can affect the accuracy of models trained on the data. To mitigate this loss, we aim to study the statistical features of data and duplicate category points, and build a model for detecting categorical duplicates in a table. 

### Structure

The data-extraction folder contains scripts in jupyter notebook to generate word pairs for our classification problem and the data itself in csv format.

The code folder contains Jupyter Notebooks to Train and Test various approaches used for the binary classification task. The dedup_lib contains the required functions and classes. 

Install the requirements according to the requirements.txt file

### Results
Performance of Random Forest Model trained on Distance features between any given pair of words. Here the pair of words can either be duplicates or non duplicates.
![image](https://user-images.githubusercontent.com/38189229/186078420-deae1549-f9a0-42fc-8d8a-71d10bbaba44.png)

### Usage

```python
# Generate Distance Features for Dataset
from dedup_lib.generateFeatureFunc import generateDistanceMetricData
DATA_PATH = <Path_To_Your_Dataset>
distFeatureDF = generateDistanceMetricData(DATA_PATH)

# Train Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['isDuplicate'], axis=1), df['isDuplicate'], test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, 
                               bootstrap = True,
                               max_features = 'sqrt')
model.fit(X_train, y_train)
```


### Relevent Papers

[1] An Empirical Study on the (Non-)Importance of Cleaning Categorical Duplicates before ML \
Vraj Shah, Thomas Parashos, and Arun Kumar

[2] Towards Benchmarking Feature Type Inference for AutoML Platforms \
Vraj Shah, Kevin Yang, and Arun Kumar
