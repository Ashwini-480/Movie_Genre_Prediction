
import joblib
import gzip
import os

# Loading the saved model
model = joblib.load("movie_genre_prediction_model.joblib")

# Compressing and saving the model
with gzip.open("movie_genre_prediction_model.joblib.gz", "wb") as f:
    joblib.dump(model, f, compress=3)

# Checking the compressed file size
size = os.path.getsize("movie_genre_prediction_model.joblib.gz") / (1024 * 1024)

print("Original model size:", round(
    os.path.getsize("movie_genre_prediction_model.joblib") / (1024 * 1024), 2
), "MB")

print("Compressed model size:", round(size, 2), "MB")
