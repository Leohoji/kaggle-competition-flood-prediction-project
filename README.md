# kaggle-competition-flood-prediction-project

This github records the notebooks of kaggle competition: [Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5/overview) from May 1, 2024 to May 31, 2024.

There are 2 tyoes of notebooks in this github:

| No | Notebook | Description |
| :- | :------- | :---------- |
| Contest | 1.[Original_Notebook_in_Kaggle_Competition](https://www.kaggle.com/code/leoloho/flood-prediction-bagging-gridsearch-0-84) 2.[Original_Notebook_in_Github](https://github.com/Leohoji/kaggle-competition-flood-prediction-project/blob/main/flood_prediction.ipynb) |  Use EDA, ensemble algorithms such as voting, bagging, and stacking, cross validation, and gridsearch to find the best model. The R2 score of best model for submission is **0.84** |
| Post-Contest | [Post_Contest_Nobook_in_Github](https://github.com/Leohoji/kaggle-competition-flood-prediction-project/blob/main/flood_prediction_post_competition.ipynb) | Use following techniques to improve R2 score: (1) feature engineering to add 8 addiotional valuable features into data; (2) implement more algorithms such as `xggoost`, `catboost`, and `lightgbm` to train the data; (3) use  **[optuna](https://optuna.readthedocs.io/zh-cn/latest/tutorial/10_key_features/005_visualization.html)** instead `GridSearchCV to search better hyperparameters, and visualize results in clear grapg. The R2 score improved in submission (late) from <ins>**0.84 to 0.86**</ins>. |


## Competition Informaiton

- **Goal**: Predict the probability of a region flooding based on various factors.
 
- **Dataset Description from competition information**: The dataset for this competition (both train and test) was generated from a deep learning model trained on the Flood Prediction Factors dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. **Note**: This dataset is particularly well suited for visualizations, clustering, and general EDA. Show off your skills

- **Evaluation**: Submissions are evaluated using the R2 score.

- **Files**: `train.csv` is the training dataset, which `FloodProbability` is the target; `test.csv` is the test dataset, your objective is to predict the FloodProbability for each row; `sample_submission.csv` is a sample submission file in the correct format.

## What techniques do I implement in this notebook?

Steps I use in [Original_Notebook](https://github.com/Leohoji/kaggle-competition-flood-prediction-project/blob/main/flood_prediction.ipynb) is folloing:

1. `Exploratory Data Analysis (EDA)`: In this step I will check the missing values, distribution, and correlation for the data.

2. `Model Training`: In this step I will first train algorithms including `voting`, `bagging`, and `stacking` in 10-percent small dataset, and then choose the best algorithm to fit the dull dataset with `GridSearchCV` technique.

3. `Prepare Sumission`: Prepare the predictions in contest-compliant format.

**Results**: Finally find the BaggingRegressor with Ridge regression as base estimator to be the best model, the R2 score is 0.84.

🎄 **What changes in [Post_Contest_Nobook_in_Github](https://github.com/Leohoji/kaggle-competition-flood-prediction-project/blob/main/flood_prediction_post_competition.ipynb)?** 🎄

Steps I use is folloing:

1. `Exploratory Data Analysis (EDA)`: In previous step I checked the missing values, distribution, and correlation for the data.

- 💪 **Improvement** 💪: I will compare the distribution of features for both training and testing data, and check whether there is outliers or abnormal value in data. Furthermore, I do feature engineering to create more valuable features into data for further improvement.

2. `Model Training`: In previous step I trained algorithms including `voting`, `bagging`, and `stacking` in 10-percent small dataset with `GridSearchCV` technique, and then choose the best algorithm to fit the dull dataset.

- 💪 **Improvement** 💪: Besides the `voting`, `bagging`, and `stacking` algorithms, I also implement `boosting` algorithms such as `XGBRegressor`, `CatBoostRegressor`, and `LGBMRegressor` to the data. Furthermore, instead of GridSearchCV, I choose the **[optuna](https://optuna.readthedocs.io/zh-cn/latest/tutorial/10_key_features/005_visualization.html)** library to search for better hyperparameters with some APIs for beautiful visualization to interpret the results. The most important of all, `cross validation` technique is implemented to my model building.

3. `Prepare Sumission`: Prepare the predictions in contest-compliant format.

**Results**:

Here is the table for all the models trained on step 2:

| Models   | Voting  | Bagging (RF) | Bagging (Linr) | Bagging (Rigde) | Bagging (Lasso) |	XGBoost | CatBoost               | 🏆 LightGBM            |	Stacking  |
| -------- | ------- | ------------ | -------------- | --------------- | --------------- | -------- | ---------------------- | ---------------------- | --------- |
| R2_score | 0.84596 | 0.86435      | 0.84605        | 0.84605         | 0.84550         | 0.86241  | <ins>**0.86718**</ins> | <ins>**0.86799**</ims> | 0.85528   |

Choose `LightGBM` to be the best model for further hyperparameter searching via optuna.

Finally, the following table is the betetr hyperparameters:

| Hyperparameters  | Value   |
| ---------------- | -----   |
| max_depth        | 8       |
| learning_rate    | 0.00879 |
| num_leaves       | 79      |
| subsample        | 0.29299 |
| colsample_bytree | 0.72709 |
| min_data_in_leaf | 98      |

The R2 score is 0.868.
