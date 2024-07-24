import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MAE, MSE, R2_SCORE = mean_absolute_error, mean_squared_error, r2_score
RMSE = lambda y_true, y_pred: np.sqrt(MSE(y_true, y_pred))

orig_features = pd.read_csv("train.csv", index_col=0).columns[1:-1].to_list()

def calculate_scores(y_true:float, y_pred:float) -> dict:
    """
    Calculate MAE, MSE, RMSE, and R2 scores.

    Args:
        y_true: True labels.
        y_pred: Model predictions
    Returns:
        Calculated results
    """
    names = ['MAE', 'MSE', 'RMSE', 'R2_score']
    calculators = [MAE, MSE, RMSE, R2_SCORE]
    results = {name: np.round(calculate(y_true, y_pred), 5) for name, calculate in zip(names, calculators)}

    return results


def create_X_y(data:pd.DataFrame, 
               full_data:bool=True) -> tuple:
    """
    Create X and y dataset from data.

    Args:
        data: Data expected to be splitted.
        full_data: Whether to split full data.
    Returns:
        Tuple of processed data.
    """
    if full_data:
        X = data.iloc[:, :-1]
        y = data["FloodProbability"]
    else:
        index = int(len(data) * 0.1)
        X = data.iloc[:index, :-1]
        y = data["FloodProbability"][:index]
    return (X, y)

def train(X_train:list, X_test:list, y_train:list, y_test:list, model) -> dict:
    """
    Train model with training and testing dataset.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data expected to be trained.
        model: Algorithm expected to fit.
    Returns:
        Dict of training results.
    """
    model = make_pipeline(StandardScaler(), model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return calculate_scores(y_test, y_pred)

def make_features(data:pd.DataFrame) -> pd.DataFrame:
    """
    Add 8 features (sum, median, std, mean, max, min, skew, kurt) into data, return processed data.

    Args:
        data: Data expected to be processed.
    Returns:
        Processed data.
    """
    df = data.copy()
    with tqdm(total=8 ,desc='Feature Extraction...') as pbar:

        df['sum'] = df.sum(axis=1)         
        pbar.update(1)
        
        df['median'] = df[orig_features].median(axis=1)         
        pbar.update(1)
        
        df['std'] = df[orig_features].std(axis=1)         
        pbar.update(1)
        
        df['mean'] = df[orig_features].mean(axis=1)
        pbar.update(1)
        
        df['max'] = df[orig_features].max(axis=1) 
        pbar.update(1)               
        
        df['min'] = df[orig_features].min(axis=1)
        pbar.update(1)
        
        df['skew'] = df[orig_features].skew(axis=1)
        pbar.update(1)
        
        df['kurt'] = df[orig_features].kurt(axis=1)
        pbar.update(1)
        
    return df

def cross_val_train(algorithm, X, y, scoring='r2', cv=3):
    """
    Cross validation of algorithm.

    Args:
        algorithm: Algorithm expected to be train.
        X: Features of data.
        y: Labels of data.
        scoring: Metrics for validation, default is 'r2'.
        cv: Number of cross validation, default is 3.
    Returns:
        The score value after training.
    """
    pipe_model = make_pipeline(StandardScaler(), algorithm)
    pipe_score = np.mean(cross_val_score(pipe_model, X, y, scoring=scoring, cv=cv, verbose=3))

    return pipe_score