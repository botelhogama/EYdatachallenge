from src.create_best_model import create_models, predict_with_saved_model

# Create the best models and save the best one
print("Creating best model...")
best_model = create_models("data/training_data_with_features_zx.csv", "model/best_model.pkl")
print("Best model created.")

