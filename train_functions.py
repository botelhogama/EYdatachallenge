import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lazypredict.Supervised import LazyRegressor

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def evaluate_models_cv(df, target_column, cv_splits=5, random_state=42, r2_threshold=0.7):
    """
    Evaluate multiple regression models using LazyRegressor with cross-validation.

    For each fold, the function trains multiple models and collects their test R² and MAPE scores.
    It then aggregates the results across folds and classifies each model as 'good'
    if the average R² meets or exceeds the r2_threshold, and 'poor' otherwise.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and the target variable.
        target_column (str): Name of the target variable column.
        cv_splits (int): Number of cross-validation folds (default 5).
        random_state (int): Seed for reproducibility (default 42).
        r2_threshold (float): Threshold to classify models as 'good' (default 0.7).

    Returns:
        aggregated_results (pd.DataFrame): A DataFrame with the average metrics (including R² and MAPE)
                                           for each model across folds, plus a 'Classification'
                                           column.
    """

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_index]
        X_test  = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test  = y.iloc[test_index]

        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mape)

        models_summary, _ = reg.fit(X_train, X_test, y_train, y_test)

        models_summary['Fold'] = fold

        fold_results.append(models_summary)

    all_results = pd.concat(fold_results)
    aggregated_results = all_results.groupby(level=0).mean()
    aggregated_results['Classification'] = aggregated_results['R-Squared'].apply(
        lambda r: 'good' if r >= r2_threshold else 'poor'
    )

    return aggregated_results
