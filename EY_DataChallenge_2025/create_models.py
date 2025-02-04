import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load your data (adjust the file path and target variable name as needed)
dataset = pd.read_csv("data/training_data_with_features.csv")
dataset = dataset.drop(columns=["datetime",
                                "longitude",
                                "latitude",
                                "Latitude",
                                "Longitude",
                                "distance",
                                "time",
                                "spatial_ref"],
                       errors='ignore')
target_column = 'UHI Index'  # Change this to your actual target column name

# Optionally, drop columns like 'latitude' and 'longitude' if needed:
dataset = dataset.drop(columns=['latitude', 'longitude'], errors='ignore')

# Split your dataset into training and testing sets
X = dataset.drop(target_column, axis=1)
y = dataset[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mapping of model names (from your LazyRegressor output) to their classes
model_mapping = {
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "BaggingRegressor": BaggingRegressor,
    "LGBMRegressor": LGBMRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}

# Define hyperparameter grids for each model (original keys)
param_grids = {
    "ExtraTreesRegressor": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    },
    "RandomForestRegressor": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    },
    "BaggingRegressor": {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
    },
    "LGBMRegressor": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
    },
    "HistGradientBoostingRegressor": {
        'max_iter': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
    }
}

# Dictionary to store the tuned models
tuned_models = {}

# Loop through each model and perform hyperparameter tuning using a pipeline
for model_name, model_class in model_mapping.items():
    print(f"\nTuning {model_name}...")
    base_param_grid = param_grids.get(model_name, {})

    # Instantiate the model (if applicable, set random_state for reproducibility)
    if model_name == "LGBMRegressor":
        model_instance = model_class(random_state=42, verbose=-1)
    else:
        try:
            model_instance = model_class(random_state=42)
        except TypeError:
            model_instance = model_class()

    # Create a pipeline: first step is scaler (default: no scaling using "passthrough"), then the model.
    pipeline = Pipeline([
        ("scaler", "passthrough"),
        ("model", model_instance)
    ])

    # Convert model parameters to pipeline parameters (prefix with "model__")
    model_param_grid = {f"model__{key}": value for key, value in base_param_grid.items()}

    # Add a parameter for the scaler step, letting grid search decide whether to use scaling or not.
    pipeline_param_grid = {"scaler": ["passthrough", StandardScaler(), MinMaxScaler()]}
    pipeline_param_grid.update(model_param_grid)

    # Set up GridSearchCV on the pipeline
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='r2',  # Optimize for R² score
        n_jobs=-1
    )

    # Fit the pipeline on the training data
    grid_search.fit(X_train, y_train)

    # Print best parameters and best cross-validated R² score
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV R² for {model_name}: {grid_search.best_score_:.4f}")

    # Save the best pipeline (including scaler choice and tuned model) for later evaluation
    tuned_models[model_name] = grid_search.best_estimator_

# Evaluate tuned models on the test set
print("\nEvaluation on Test Data:")
for name, pipeline in tuned_models.items():
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mape_value = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
    print(f"{name} -- Test R²: {r2:.4f}, Test MAPE: {mape_value:.2f}%")

