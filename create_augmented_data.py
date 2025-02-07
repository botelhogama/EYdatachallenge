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

# For robust imputation
from sklearn.impute import KNNImputer

#########################
# 1. Load Training Data #
#########################

print("Loading training data...")
training_data = pd.read_csv("data/training_data_uhi_index.csv")
training_data.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)
print("Training data loaded successfully.")

##############################
# 2. STAC Search and Load Data #
##############################

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

###########################
# 3. Data Conversion      #
###########################

print("Converting data to DataFrame...")
df = data.to_dataframe().reset_index()

# Convert 'time' to datetime
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

# Ensure training_data 'datetime' is also in datetime format
training_data['datetime'] = pd.to_datetime(training_data['datetime'], format='%d-%m-%Y %H:%M')

print("Data conversion completed.")

#########################################
# 4. Filter Data by Training Time Range #
#########################################

print("Filtering data within the training time range...")
df_filtered = df[
    (df['time'] <= training_data['datetime'].max()) &
    (df['time'] >= training_data['datetime'].min())
]
print(f"Filtered data contains {df_filtered.shape[0]} rows and {df_filtered.shape[1]} columns.")

if not os.path.exists('data/df_filtered_date_step1.csv'):
    df_filtered.to_csv('data/df_filtered_date_step1.csv', index=False)

#####################################
# 5. Compute Spectral Indices       #
#####################################

# Use a small constant to avoid division by zero
epsilon = 1e-6

# NDVI (Normalized Difference Vegetation Index)
ndvi = (df_filtered.B08 - df_filtered.B04) / (df_filtered.B08 + df_filtered.B04 + epsilon)

# MNDWI (Modified Normalized Difference Water Index)
mndwi = (df_filtered.B03 - df_filtered.B11) / (df_filtered.B03 + df_filtered.B11 + epsilon)

# NDBI (Normalized Difference Built-up Index)
ndbi = (df_filtered.B11 - df_filtered.B08) / (df_filtered.B11 + df_filtered.B08 + epsilon)


# SAVI (Soil-Adjusted Vegetation Index) with L = 0.5
L = 0.5
savi = ((df_filtered.B08 - df_filtered.B04) * (1 + L)) / (df_filtered.B08 + df_filtered.B04 + L + epsilon)

# NDMI (Normalized Difference Moisture Index)
ndmi = (df_filtered.B08 - df_filtered.B11) / (df_filtered.B08 + df_filtered.B11 + epsilon)

# NBR (Normalized Burn Ratio)
nbr = (df_filtered.B08 - df_filtered.B12) / (df_filtered.B08 + df_filtered.B12 + epsilon)

# Add all indices to the dataset
df_filtered = df_filtered.assign(
    NDVI=ndvi,
    MNDWI=mndwi,
    NDBI=ndbi,
    SAVI=savi,
    NDMI=ndmi,
    NBR=nbr
)
print("All spectral indices (NDVI, MNDWI, NDBI, SAVI, NDMI, NBR) added to dataset.")

if not os.path.exists('data/df_NDVI_MNDWI.csv'):
    df_filtered.to_csv('data/df_NDVI_MNDWI.csv', index=False)

##################################
# 6. Feature Extraction          #
##################################

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
    lat, lon = row["latitude"], row["longitude"]
    features.append(extract_features_from_dataframe(lat, lon, df_filtered))

# Combine extracted features with training_data
extracted_features = pd.DataFrame(features)
training_data_with_features = pd.concat([training_data.reset_index(drop=True),
                                         extracted_features.reset_index(drop=True)], axis=1)

###############################################
# 7. Handle Missing Values with KNN Imputer   #
###############################################

# Create a copy of the final DataFrame for processing
df_final = training_data_with_features.copy()

# Calculate the fraction of nulls per column
null_fraction = df_final.isnull().mean()
print("Missing value percentages per column:")
print(null_fraction)

# Identify columns with more than 20% missing values and drop them
cols_to_drop = null_fraction[null_fraction > 0.2].index.tolist()
print(f"Dropping columns (>{20}% missing): {cols_to_drop}")
df_final = df_final.drop(columns=cols_to_drop)

# Identify numeric columns for imputation
numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

# Identify numeric columns that still have missing values
cols_to_impute = [col for col in numeric_cols if df_final[col].isnull().sum() > 0]
print(f"Imputing missing values in columns: {cols_to_impute}")

# Initialize and apply the KNN imputer on numeric columns
imputer = KNNImputer(n_neighbors=5)
df_final[numeric_cols] = imputer.fit_transform(df_final[numeric_cols])

# Check again for missing values
print("Missing values after imputation:")
print(df_final.isnull().sum())

# Replace the original DataFrame with the processed one
training_data_with_features = df_final.copy()

###############################################
# 8. Save Final Training Data with Features   #
###############################################

print("Saving training data with extracted and imputed features...")
if not os.path.exists("data/training_data_with_features_zx.csv"):
    training_data_with_features.to_csv("data/training_data_with_features_zx.csv", index=False)
print("Processing completed successfully. File saved.")
