import os
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Import common GIS tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
import pandas as pd
from matplotlib.cm import RdYlGn, jet, RdBu

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer
from odc.stac import stac_load

print("Loading training data...")
training_data = pd.read_csv("data/training_data_uhi_index.csv")
training_data.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)
print("Training data loaded successfully.")

# Define search bounds
lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)
time_window = "2021-06-01/2021-09-01"
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

print("Initializing STAC client...")
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

print("Searching for Sentinel-2 imagery...")
search = stac.search(
    bbox=bounds,
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)
items = list(search.get_items())
signed_items = [planetary_computer.sign(item).to_dict() for item in items]
print(f"Found {len(items)} items matching the search criteria.")

# Load data
resolution = 10  # meters per pixel
scale = resolution / 111320.0  # degrees per pixel for crs=4326

print("Loading remote sensing data...")
data = stac_load(
    items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326",  # Latitude-Longitude
    resolution=scale,  # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds
)
print("Remote sensing data successfully loaded.")

print("Converting data to DataFrame...")
df = data.to_dataframe().reset_index()

# Converte 'time' para datetime sem transformá-lo em string
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

# Certifique-se de que training_data['datetime'] também está em datetime
training_data['datetime'] = pd.to_datetime(training_data['datetime'], format='%d-%m-%Y %H:%M')

print("Data conversion completed.")

print("Filtering data within the training time range...")
df_filtered = df[
    (df['time'] <= training_data['datetime'].max()) &
    (df['time'] >= training_data['datetime'].min())
]
print(f"Filtered data contains {df_filtered.shape[0]} rows and contains {df_filtered.shape[1]} columns.")

df_filtered.to_csv('data/df_filtered_date_step1.csv', index=False) if not os.path.exists('data/df_filtered_date_step1.csv') else None

# Compute NDVI and MNDWI
print("Computing NDVI and MNDWI...")
ndvi = (df_filtered.B08 - df_filtered.B04) / (df_filtered.B08 + df_filtered.B04)
mndwi = (df_filtered.B03 - df_filtered.B11) / (df_filtered.B03 + df_filtered.B11)

# Add NDVI and MNDWI to the dataset
df_filtered = df_filtered.assign(NDVI=ndvi, MNDWI=mndwi)
print("NDVI and MNDWI added to dataset.")

df_filtered.to_csv('data/df_NDVI_MNDWI.csv', index=False) if not os.path.exists('data/df_NDVI_MNDWI.csv') else None

# Function to extract features for a point from a pandas DataFrame
def extract_features_from_dataframe(lat, lon, df_filtered):
    # Calculate distances between the point and all rows in df_filtered
    df_filtered["distance"] = np.sqrt(
        (df_filtered["Latitude"] - lat) ** 2 + (df_filtered["Longitude"] - lon) ** 2
    )
    # Find the closest point
    closest_row = df_filtered.loc[df_filtered["distance"].idxmin()]
    # Return the feature values as a dictionary
    return closest_row.to_dict()

# Rename columns to match if needed
df_filtered.rename(columns={"latitude": "Latitude", "longitude": "Longitude"}, inplace=True)

# Iterate over training_data and extract features
features = []
for _, row in training_data.iterrows():
    lat, lon = row["Latitude"], row["Longitude"]
    features.append(extract_features_from_dataframe(lat, lon, df_filtered))

# Combine extracted features with training_data
extracted_features = pd.DataFrame(features)
training_data_with_features = pd.concat([training_data.reset_index(drop=True), extracted_features.reset_index(drop=True)], axis=1)

print("Saving training data with extracted features...")
training_data_with_features.to_csv("data/training_data_with_features.csv", index=False) if not os.path.exists("data/training_data_with_features.csv") else None
print("Processing completed successfully. File saved.")
