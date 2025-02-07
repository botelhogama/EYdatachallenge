import pandas as pd
from src.train_functions import evaluate_models_cv


dataset = pd.read_csv("data/combined_cities_features.csv")
dataset = dataset.drop(columns=["datetime",
                                "longitude",
                                "latitude",
                                "Latitude",
                                "Longitude",
                                "distance",
                                "time",
                                "spatial_ref"],
                       errors='ignore')
results = evaluate_models_cv(dataset, "UHI Index")
results = results.sort_values(by='R-Squared', ascending=False)
results.to_csv("data/evaluation_models.csv", index=True)