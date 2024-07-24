# kaggle-competition-flood-prediction-project

This github records the notebooks of kaggle competition: [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5/overview). 

Here is the informaiton about this competition:

**Time**: May 1, 2024 to May 31, 2024.

**Competition Goal**: Predict the probability of a region flooding based on various factors.

**Dataset Description from competition information**

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Flood Prediction Factors dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Note: This dataset is particularly well suited for visualizations, clustering, and general EDA. Show off your skills

**Evaluation**

Submissions are evaluated using the R2 score.

**Files**

- `train.csv`: The training dataset, `FloodProbability` is the target.

- `test.csv`: The test dataset, your objective is to predict the FloodProbability for each row.

- `sample_submission.csv`: A sample submission file in the correct format. format


**What techniques do I implement in this notebook?**

There are 2 notebooks in this github:

| Notebook | Description | Techniques | R2 Score |
| -------- | ----------- | ---------- | -------- |
|  [Original_Notebook_in_Kaggle_Competition](https://www.kaggle.com/code/leoloho/flood-prediction-bagging-gridsearch-0-84) & [Original_Notebook_in_Github]| | | |


Steps I use is folloing:

1. `Exploratory Data Analysis (EDA)`: In this step I will check the missing values, distribution, and correlation for the data.

2. `Model Training`: In this step I will first train algorithms including `voting`, `bagging`, and `stacking` in 10-percent small dataset, and then choose the best algorithm to fit the dull dataset with `GridSearchCV` technique.

3. `Prepare Sumission`: Prepare the predictions in contest-compliant format.


🎄 **What changes in this notebook?** 🎄

✨ Here is the original notebook in kaggle: **[original notebook in kaggle competition](https://www.kaggle.com/code/leoloho/flood-prediction-bagging-gridsearch-0-84)**, and the idea of improvement is inspired by this **[notebook](https://www.kaggle.com/code/arunl15/ensemble-model-xgb-lgbm-cbr-xgbrf-ridge#Feature-Engineering)** from other competitioner.

Steps I use is folloing:

1. `Exploratory Data Analysis (EDA)`: In previous step I checked the missing values, distribution, and correlation for the data.

- 💪 **Improvement** 💪: I will compare the distribution of features for both training and testing data, and check whether there is outliers or abnormal value in data. Furthermore, I do feature engineering to create more valuable features into data for further improvement.

2. `Model Training`: In previous step I trained algorithms including `voting`, `bagging`, and `stacking` in 10-percent small dataset with `GridSearchCV` technique, and then choose the best algorithm to fit the dull dataset.

- 💪 **Improvement** 💪: Besides the `voting`, `bagging`, and `stacking` algorithms, I also implement `boosting` algorithms such as `XGBRegressor`, `CatBoostRegressor`, and `LGBMRegressor` to the data. Furthermore, instead of GridSearchCV, I choose the **[optuna](https://optuna.readthedocs.io/zh-cn/latest/tutorial/10_key_features/005_visualization.html)** library to search for better hyperparameters with some APIs for beautiful visualization to interpret the results. The most important of all, `cross validation` technique is implemented to my model building.

3. `Prepare Sumission`: Prepare the predictions in contest-compliant format.


| Models   | Voting  | Bagging (RF) | Bagging (Linr) | Bagging (Rigde) | Bagging (Lasso) |	XGBoost | CatBoost               | LightGBM               |	Stacking  |
| -------- | ------- | ------------ | -------------- | --------------- | --------------- | -------- | ---------------------- | ---------------------- | --------- |
| R2_score | 0.84596 | 0.86435      | 0.84605        | 0.84605         | 0.84550         | 0.86241  | <ins>**0.86718**</ins> | <ins>**0.86799**</ims> | 0.85528   |
