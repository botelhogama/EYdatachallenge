from src.create_best_model import predict_with_saved_model

print("Predicting with best model...")
pred_df = predict_with_saved_model('model/best_model.pkl', "data/submission_with_features.csv")
print("Predictions saved.")