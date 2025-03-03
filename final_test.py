#%%
import os
# Supress Warnings
import warnings

from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import common GIS tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
import pandas as pd
from matplotlib.cm import RdYlGn,jet,RdBu

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer
from odc.stac import stac_load
#%%
# Calculate NDVI
training_data = pd.read_csv("../data_test/training_data_uhi_index.csv")
print(training_data.columns)
training_data['datetime'] = pd.to_datetime(training_data['datetime'], format='%d-%m-%Y %H:%M')
training_data.describe()
#%%
# Calculate the bounds for doing an archive data search
# bounds = (min_lon, min_lat, max_lon, max_lat)
lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
time_window = "2021-07-23/2021-07-25"
height = 100
width = 100
#%%
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = stac.search(
    bbox=bounds,
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)
#%%
items = list(search.get_items())
print('This is the number of scenes that touch our region:',len(items))
signed_items = [planetary_computer.sign(item).to_dict() for item in items]
#%%
resolution = 10  # meters per pixel
scale = resolution / 111320.0 # degrees per pixel for crs=4326
#%%
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

#%%
df = train_feat = data.to_dataframe().reset_index()
print(df.head())

df['time'] = pd.to_datetime(df['time'])
df['time'] = df['time'].dt.strftime('%d-%m-%Y %H:%M')
display(df)
training_data['datetime'] = pd.to_datetime(training_data['datetime'], format='%d-%m-%Y %H:%M')
#%%
training_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
#%%
training
#%%
# Function to extract features for a point from a pandas DataFrame
def extract_features_from_dataframe(lat, lon, df):
    # Calculate distances between the point and all rows in df
    df["distance"] = np.sqrt(
        (df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2
    )
    # Find the closest point
    closest_row = df.loc[df["distance"].idxmin()]
    # Return the feature values as a dictionary
    return closest_row.to_dict()

# Iterate over training_data and extract features
features = []
for _, row in training_data.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    features.append(extract_features_from_dataframe(lat, lon, df))

# Combine extracted features with training_data
extracted_features = pd.DataFrame(features)
training_data_with_features = pd.concat([training_data.reset_index(drop=True), extracted_features.reset_index(drop=True)], axis=1)
training_data_with_features = training_data_with_features.loc[:, ~training_data_with_features.columns.duplicated()]

conversion_factor = 111320  # metros por grau (aproximadamente no equador)
training_data_with_features['distance_meters'] = training_data_with_features['distance'] * conversion_factor



#%%
# Exemplo: Criação de NDVI e SAVI
training_data_with_features['NDVI'] = (training_data_with_features['B08'] - training_data_with_features['B04']) / (training_data_with_features['B08'] + training_data_with_features['B04'] + 1e-6)
L = 0.5
training_data_with_features['SAVI'] = ((training_data_with_features['B08'] - training_data_with_features['B04']) * (1 + L)) / (training_data_with_features['B08'] + training_data_with_features['B04'] + L + 1e-6)

# Criação de NDBI
training_data_with_features['NDBI'] = (training_data_with_features['B11'] - training_data_with_features['B08']) / (training_data_with_features['B11'] + training_data_with_features['B08'] + 1e-6)

# Criação de MNDWI
training_data_with_features['MNDWI'] = (training_data_with_features['B03'] - training_data_with_features['B11']) / (training_data_with_features['B03'] + training_data_with_features['B11'] + 1e-6)

# Criação de EVI
training_data_with_features['EVI'] = 2.5 * (training_data_with_features['B08'] - training_data_with_features['B04']) / (training_data_with_features['B08'] + 6 * training_data_with_features['B04'] - 7.5 * training_data_with_features['B02'] + 1)
#%%
print(training_data_with_features.columns)
training_data_with_features = training_data_with_features.drop(columns=['datetime', 'time'], axis=1, errors = 'ignore')
print(training_data_with_features.std())
#%%
training_data_with_features_x = training_data_with_features
#%%
import matplotlib.pyplot as plt
import numpy as np

# Remove unnecessary columns
df1 = training_data_with_features.drop(columns=["datetime", "time", "spatial_ref", "EVI"],
                                         errors='ignore')
df_plot = df1.copy()

# Get the list of features (columns) to plot
features = df_plot.columns.tolist()
n_features = len(features)

# Set number of columns per row for the subplots
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Create a figure and subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
axes = axes.flatten()  # Flatten axes in case it's a 2D array

for i, feature in enumerate(features):
    series = df_plot[feature].dropna()

    # If the first element is iterable (but not a string), flatten the series
    if series.size > 0:
        sample = series.iloc[0]
        if hasattr(sample, '__iter__') and not isinstance(sample, str):
            data = np.concatenate(series.values)
        else:
            data = series.values
    else:
        data = series.values

    # Plot histogram for the feature
    axes[i].hist(data, bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")

# Hide any empty subplots (if the number of features is odd)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Remover colunas indesejadas para manter apenas as features desejadas
df1 = training_data_with_features.drop(columns=["datetime",
                        "time",
                        "spatial_ref",
                        "EVI"],
                       errors='ignore')

# Utilize df1 (ou outro DataFrame final com as features) para plotar
df_plot = df1.copy()

# Obter a lista de features
features = df_plot.columns.tolist()
n_features = len(features)

# Definir número de colunas por linha para os subplots
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Criar figura e subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
axes = axes.flatten()  # Se os eixos estiverem em array 2D, transforma em 1D

# Loop para criar o boxplot de cada feature
for i, feature in enumerate(features):
    data = df_plot[feature].dropna()  # Remover NaNs para um boxplot mais limpo
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='skyblue', color='black'),
                    medianprops=dict(color='red'))
    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Valores")

# Esconder subplots vazios (caso o número de features seja ímpar)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

#%%
## corr matrix heat map
import seaborn as sns

# Remover colunas indesejadas para manter apenas as features desejadas
df1 = training_data_with_features.drop(columns=["datetime",
                        "time",
                        "spatial_ref",
                        "EVI"],
                       errors='ignore')

# Utilize df1 (ou outro DataFrame final com as features) para plotar
df_plot = df1.copy()

# Calcular a matriz de correlação spearman
corr = df_plot.corr(method='spearman')

# Criar um mapa de calor da matriz de correlação
plt.figure(figsize=(12, 10))

# Definir o mapa de cores
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Plotar o mapa de calor
sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .5})
#%%
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, factor=3):
    print(df)
    df.reset_index(drop=True)
    df_clean = df.copy()
    # Get list of numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Keep rows within the bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Example usage:
# Assuming df is your DataFrame
df_clean_factor3 = remove_outliers_iqr(df1, factor=3)
df_clean_factor4 = remove_outliers_iqr(df1, factor=4)
df_clean_factor5 = remove_outliers_iqr(df1, factor=5)
df_clean_factor2 = remove_outliers_iqr(df1, factor=2)
print("Data shape before outlier removal:", df1.shape)
print("Data shape after outlier removal:", df_clean_factor3.shape)

#%%
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Necessário para ativar o IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import KNNImputer


def replace_outliers_with_nan(df, factor=3):
    # Cria uma cópia do DataFrame para não modificar o original
    df_clean = df.copy()
    # Seleciona as colunas numéricas
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Substitui por NaN os valores que estão fora do intervalo [lower_bound, upper_bound]
        df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan

    return df_clean

# Exemplo de uso:
# Suponha que df1 seja seu DataFrame original
df_transformed_imputer = replace_outliers_with_nan(df1, factor=3)
print("Formato dos dados antes e depois (deve permanecer o mesmo):", df1.shape, df_transformed_imputer.shape)
print("Número de NaNs por coluna após a transformação:")
print(df_transformed_imputer.isna().sum())

mice_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
df_imputed_mice = pd.DataFrame(mice_imputer.fit_transform(df_transformed_imputer), columns=df_transformed_imputer.columns)
knn_imputer = KNNImputer(n_neighbors=10)
df_imputed_knn = pd.DataFrame(knn_imputer.fit_transform(df_transformed_imputer), columns=df_transformed_imputer.columns)
#%%
## corr matrix heat map
import seaborn as sns

# Utilize df1 (ou outro DataFrame final com as features) para plotar
df_plot = df_clean_factor3.copy()

# Calcular a matriz de correlação spearman
corr = df_plot.corr(method='spearman')

# Criar um mapa de calor da matriz de correlação
plt.figure(figsize=(12, 10))

# Definir o mapa de cores
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Plotar o mapa de calor
sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .5})
#%%
import matplotlib.pyplot as plt
import numpy as np

df_plot = df_clean_factor3.copy()

# Obter a lista de features
features = df_plot.columns.tolist()
n_features = len(features)

# Definir número de colunas por linha para os subplots
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Criar figura e subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
axes = axes.flatten()  # Se os eixos estiverem em array 2D, transforma em 1D

# Loop para criar o boxplot de cada feature
for i, feature in enumerate(features):
    data = df_plot[feature].dropna()  # Remover NaNs para um boxplot mais limpo
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='skyblue', color='black'),
                    medianprops=dict(color='red'))
    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Valores")

# Esconder subplots vazios (caso o número de features seja ímpar)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, factor=3):
    df_clean = df.copy()
    # Get list of numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Keep rows within the bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# Example usage:
# Assuming df is your DataFrame
df_clean_factor5 = remove_outliers_iqr(df1, factor=1.5)
print("Data shape before outlier removal:", df1.shape)
print("Data shape after outlier removal:", df_clean_factor5.shape)

#%%
import matplotlib.pyplot as plt
import numpy as np

df_plot = df_clean_factor5.copy()

# Obter a lista de features
features = df_plot.columns.tolist()
n_features = len(features)

# Definir número de colunas por linha para os subplots
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Criar figura e subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
axes = axes.flatten()  # Se os eixos estiverem em array 2D, transforma em 1D

# Loop para criar o boxplot de cada feature
for i, feature in enumerate(features):
    data = df_plot[feature].dropna()  # Remover NaNs para um boxplot mais limpo
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='skyblue', color='black'),
                    medianprops=dict(color='red'))
    axes[i].set_title(feature)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Valores")

# Esconder subplots vazios (caso o número de features seja ímpar)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

#%%
## corr matrix heat map
import seaborn as sns

# Utilize df1 (ou outro DataFrame final com as features) para plotar
df_plot = df_clean_factor5.copy()

# Calcular a matriz de correlação spearman
corr = df_plot.corr(method='spearman')

# Criar um mapa de calor da matriz de correlação
plt.figure(figsize=(12, 10))

# Definir o mapa de cores
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Plotar o mapa de calor
sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .5})
#%%
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

# ------------------------------
# 1. Carregar Dados e Pré-processamento
# ------------------------------
# Carregue seu dataset (ajuste o caminho conforme necessário)

# data = training_data_with_features
# data = df_clean_factor5.copy()
data = df_clean_factor3.copy()
# data = df_imputed_mice
# data = df_imputed_knn
# data = df_clean_factor4.copy()
# data = df_clean_factor5.copy()
# data = df_clean_factor2.copy()

# Remova as colunas irrelevantes
cols_to_drop = ["latitude", "longitude","datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Defina a variável target e remova colunas duplicadas de coordenadas, se houver
target = "UHI Index"

X = data.drop(target, axis=1)
y = data[target]

# Divida os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ------------------------------
# 2. Construção do Pipeline
# ------------------------------
# Pipeline com:
# - VarianceThreshold: remove features com variância < 0.01
# - SelectFromModel: seleciona features com importância acima da média (usando ExtraTreesRegressor)
# - SelectKBest: escolhe as 5 melhores features com f_regression
# - StandardScaler: padroniza os dados
# - Modelo: ExtraTreesRegressor com hiperparâmetros otimizados
pipeline = Pipeline([
    # ("variance_threshold", VarianceThreshold(threshold=0.01)),
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.99*mean"
    )),
    ("scaler", RobustScaler()),
    # ("select_kbest", SelectKBest(score_func=f_regression, k=5)),
    ("model", ExtraTreesRegressor(random_state=42, n_estimators=600, max_depth=100, min_samples_split=2))
])

# ------------------------------
# 3. Treinamento e Avaliação
# ------------------------------
print("Treinando o pipeline...")
pipeline.fit(X_train, y_train)

# selected_mask = pipeline.named_steps['select_from_model'].get_support()
# selected_features = X_train.columns[selected_mask]
# print("Features usadas na predição:", selected_features.tolist())

# Avaliação no conjunto de teste
y_pred = pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

pipeline.fit(X, y)
#%%
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assuming df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build the Pipeline with Polynomial Features
# ------------------------------
# Pipeline includes:
# - SelectFromModel: select features with importance above threshold
# - PolynomialFeatures: generate interaction and squared features (degree 2)
# - RobustScaler: scale the data
# - Model: ExtraTreesRegressor with optimized hyperparameters
pipeline = Pipeline([
    # You can include VarianceThreshold if needed, here we keep just SelectFromModel:
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.99*mean"
    )),
    # Add polynomial features (degree=2) without adding a constant term (bias)
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", RobustScaler()),
    ("model", ExtraTreesRegressor(random_state=42, n_estimators=600, max_depth=100, min_samples_split=2))
])

# ------------------------------
# 3. Train and Evaluate the Pipeline
# ------------------------------
print("Training the pipeline with Polynomial Features...")
pipeline.fit(X_train, y_train)

# If you want to inspect the selected features from the select_from_model step:
selected_mask = pipeline.named_steps['select_from_model'].get_support()
selected_features = X_train.columns[selected_mask]
print("Features used in prediction (before poly expansion):", selected_features.tolist())

# Evaluate on test set
y_pred = pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

pipeline.fit(X, y)
# ------------------------------
# Optionally, save the trained pipeline
# ------------------------------
model_filename = "trained_pipeline_poly.pkl"
joblib.dump(pipeline, model_filename)
print("Pipeline saved as:", model_filename)

#%%
model_filename = "trained_model_pipeline.pkl"
joblib.dump(pipeline, model_filename)
print("Pipeline salvo como '{}'".format(model_filename))
#%%
df_clean_factor5.columns
#%%
sub_temp = pd.read_csv('../data/Submission_template.csv')
sub_temp.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)
features = []
for _, row in sub_temp.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    features.append(extract_features_from_dataframe(lat, lon, train_feat))

val_features = pd.DataFrame(features)
val_data_with_features = pd.concat([sub_temp.reset_index(drop=True), val_features.reset_index(drop=True)], axis=1)
val_data_with_features = val_data_with_features.loc[:, ~val_data_with_features.columns.duplicated()]
val_data_with_features
#%%
# Exemplo: Criação de NDVI e SAVI
val_data_with_features['NDVI'] = (val_data_with_features['B08'] - val_data_with_features['B04']) / (val_data_with_features['B08'] + val_data_with_features['B04'] + 1e-6)
L = 0.5
val_data_with_features['SAVI'] = ((val_data_with_features['B08'] - val_data_with_features['B04']) * (1 + L)) / (val_data_with_features['B08'] + val_data_with_features['B04'] + L + 1e-6)

# Criação de NDBI
val_data_with_features['NDBI'] = (val_data_with_features['B11'] - val_data_with_features['B08']) / (val_data_with_features['B11'] + val_data_with_features['B08'] + 1e-6)

# Criação de MNDWI
val_data_with_features['MNDWI'] = (val_data_with_features['B03'] - val_data_with_features['B11']) / (val_data_with_features['B03'] + val_data_with_features['B11'] + 1e-6)

# Criação de EVI
val_data_with_features['EVI'] = 2.5 * (val_data_with_features['B08'] - val_data_with_features['B04']) / (val_data_with_features['B08'] + 6 * val_data_with_features['B04'] - 7.5 * val_data_with_features['B02'] + 1)
#%%
# val_data_with_features = val_data_with_features[['longitude', 'latitude', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06','B07', 'B08', 'B8A', 'B11', 'B12']]
copy_val_data = val_data_with_features[['B01', 'B02', 'B03', 'B04', 'B05', 'B06','B07', 'B08', 'B8A', 'B11', 'B12', 'NDVI', 'SAVI', 'NDBI', 'MNDWI']].copy()
pred_vals = pipeline.predict(copy_val_data)
#%%
pred_vals
#%%
data_to_send = pd.DataFrame()
data_to_send['UHI Index'] = pred_vals
data_to_send['Latitude'] = val_data_with_features['latitude']
data_to_send['Longitude'] = val_data_with_features['longitude']

data_to_send = data_to_send[['Longitude', 'Latitude', 'UHI Index']]
data_to_send.to_csv('../outputs/poly_predicted_values.csv', index=False)
#%%
## need an histogram of np.array
import matplotlib.pyplot as plt
import numpy as np

# Create a histogram of the predicted values
plt.hist(pred_vals, bins=30, color='skyblue', edgecolor='black')

#%%
data_to_send = pd.DataFrame()
data_to_send['UHI Index'] = pred_vals
data_to_send['Latitude'] = val_data_with_features['latitude']
data_to_send['Longitude'] = val_data_with_features['longitude']

data_to_send = data_to_send[['Longitude', 'Latitude', 'UHI Index']]
data_to_send.to_csv('../outputs/predicted_values.csv', index=False)
#%% md
# #TESTES!!!
#%%
temperature_data = pd.read_csv('../data/hyperlocal_temperature_monitoring.csv')
temperature_data
#%%
temperature_data['day'] = pd.to_datetime(temperature_data['day'])
temperature_data['day'] = temperature_data['day'].dt.date
mean_airtemp_by_location = temperature_data.groupby(['day', 'latitude', 'longitude'])['airtemp'].mean().reset_index()

print(mean_airtemp_by_location)

#%%
mean_airtemp_by_location['airtemp'].hist()
#%%


# Function to extract features for a point from a pandas DataFrame
def extract_features_from_dataframe(lat, lon, df):
    # Calculate distances between the point and all rows in df
    df["distance"] = np.sqrt(
        (df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2
    )
    # Find the closest point
    closest_row = df.loc[df["distance"].idxmin()]
    # Return the feature values as a dictionary
    return closest_row.to_dict()

# Iterate over training_data and extract features
features = []
for _, row in training_data.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    features.append(extract_features_from_dataframe(lat, lon, mean_airtemp_by_location))

features
# Combine extracted features with training_data
extracted_temperature = pd.DataFrame(features)
training_data_with_features = pd.concat([training_data_with_features_x.reset_index(drop=True), extracted_temperature.reset_index(drop=True)], axis=1)

training_data_with_features = training_data_with_features.loc[:, ~training_data_with_features.columns.duplicated()]
training_data_with_features


training_data_with_features.to_csv('../data/training_data_with_features_corr.csv', index=False)
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suponha que training_data_with_features já esteja definido
# Exemplo de impressão das colunas:
print("Colunas disponíveis:", training_data_with_features.columns.tolist())

# Crie uma cópia do DataFrame para o cálculo da correlação
df_plot = training_data_with_features.copy()

# Remova as colunas não numéricas ou que não deseja incluir na correlação
cols_to_drop = ['datetime', 'time', 'spatial_ref', 'EVI', 'day']
df_plot = df_plot.drop(columns=cols_to_drop, errors='ignore')

# Converta todas as colunas para valores numéricos (valores não convertíveis se tornam NaN)
df_plot = df_plot.apply(pd.to_numeric, errors='coerce')

# Calcule a matriz de correlação usando o método Spearman
corr = df_plot.corr(method='spearman')
print("Matriz de correlação:")
print(corr)

# Crie um mapa de calor da matriz de correlação
plt.figure(figsize=(12, 10))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Mapa de Calor da Correlação (Spearman)")
plt.show()

#%%
df_clean_factor_temp = remove_outliers_iqr(training_data_with_features, factor=3)
df_clean_factor_temp
#%%
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

# ------------------------------
# 1. Carregar Dados e Pré-processamento
# ------------------------------
# Carregue seu dataset (ajuste o caminho conforme necessário)

# data = training_data_with_features
# data = df_clean_factor5.copy()
# data = df_clean_factor3.copy()
data = df_clean_factor_temp.copy()
# data = df_imputed_mice
# data = df_imputed_knn

# Remova as colunas irrelevantes
cols_to_drop = ["latitude",
                "longitude",
                "datetime",
                "distance",
                "distance_meters",
                "time",
                "spatial_ref",
                "EVI",
                'sensor_id',
                'location',
                'distance_to_center',
                'year',
                'hour',
                'install_type',
                'ntacode',
                'borough',
                'day']
data = data.drop(columns=cols_to_drop, errors="ignore")

# Defina a variável target e remova colunas duplicadas de coordenadas, se houver
target = "UHI Index"

X = data.drop(target, axis=1)
y = data[target]
print(X.columns)
# Divida os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ------------------------------
# 2. Construção do Pipeline
# ------------------------------
# Pipeline com:
# - VarianceThreshold: remove features com variância < 0.01
# - SelectFromModel: seleciona features com importância acima da média (usando ExtraTreesRegressor)
# - SelectKBest: escolhe as 5 melhores features com f_regression
# - StandardScaler: padroniza os dados
# - Modelo: ExtraTreesRegressor com hiperparâmetros otimizados
pipeline = Pipeline([
    # ("variance_threshold", VarianceThreshold(threshold=0.01)),
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.98*mean"
    )),
    ("scaler", RobustScaler()),
    # ("select_kbest", SelectKBest(score_func=f_regression, k=4)),
    ("model", ExtraTreesRegressor(random_state=42, n_estimators=600, max_depth=300, min_samples_split=2))
])

# ------------------------------
# 3. Treinamento e Avaliação
# ------------------------------
print("Treinando o pipeline...")
pipeline.fit(X_train, y_train)

selected_mask = pipeline.named_steps['select_from_model'].get_support()
selected_features = X_train.columns[selected_mask]
print("Features usadas na predição:", selected_features.tolist())

# Avaliação no conjunto de teste
y_pred = pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

#%%
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assuming df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build the Pipeline with Imputation and Polynomial Features
# ------------------------------
pipeline = Pipeline([
    # Use SelectFromModel to filter features based on importance
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.99*mean"
    )),
    # Add a SimpleImputer to fill any missing values (median strategy)
    ("imputer", SimpleImputer(strategy="median")),
    # Generate polynomial features (degree 3, without bias)
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    # Scale the features robustly
    ("scaler", RobustScaler()),
    # Final model: ExtraTreesRegressor with specified hyperparameters
    ("model", ExtraTreesRegressor(random_state=42, n_estimators=600, max_depth=100, min_samples_split=2))
])

# ------------------------------
# 3. Train and Evaluate the Pipeline
# ------------------------------
print("Training the pipeline with enhanced feature engineering and imputation...")
pipeline.fit(X_train, y_train)

# Optionally inspect features used before polynomial expansion
selected_mask = pipeline.named_steps['select_from_model'].get_support()
selected_features = X_train.columns[selected_mask]
print("Features used in prediction (before poly expansion):", selected_features.tolist())

# Evaluate on test set
y_pred = pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# Refit the pipeline on the entire dataset
pipeline.fit(X, y)

# ------------------------------
# 4. Save the Trained Pipeline
# ------------------------------
model_filename = "trained_pipeline_poly.pkl"
joblib.dump(pipeline, model_filename)
print("Pipeline saved as:", model_filename)

#%%
val_data_with_features
#%%
val_data_feat = val_data_with_features
sub_temp = pd.read_csv('../data/Submission_template.csv')
sub_temp.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)
features = []


for _, row in val_data_with_features.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    features.append(extract_features_from_dataframe(lat, lon, mean_airtemp_by_location))

temp_val_features = pd.DataFrame(features)
val_data_feat = pd.concat([val_data_feat.reset_index(drop=True), temp_val_features.reset_index(drop=True)], axis=1)
val_data_feat = val_data_feat.loc[:, ~val_data_feat.columns.duplicated()]
val_data_feat
#%%
val_data_with_features['airtemp'].hist()
#%%
# val_data_with_features = val_data_with_features[['longitude', 'latitude', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06','B07', 'B08', 'B8A', 'B11', 'B12']]
copy_val_data = val_data_with_features[['airtemp', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
       'B8A', 'B11', 'B12']].copy()
pred_vals = pipeline.predict(copy_val_data)
#%%
data_to_send = pd.DataFrame()
data_to_send['UHI Index'] = pred_vals
data_to_send['Latitude'] = sub_temp['latitude']
data_to_send['Longitude'] = sub_temp['longitude']

data_to_send = data_to_send[['Longitude', 'Latitude', 'UHI Index']]
data_to_send.to_csv('../outputs/predicted_values_airtemp.csv', index=False)
#%%
data_to_send
#%%
plt.hist(data_to_send['UHI Index'], bins=30, color='skyblue', edgecolor='black')
#%%
global_data = pd.read_csv('../data/GlobalLandTemperaturesByCity.csv')
#%%
global_data_1 = global_data[global_data['City'] == 'New York']
#%%
global_data_1
#%%
poly_pred_vals = pd.read_csv('../outputs/poly_predicted_values.csv')
tree_pred_vals = pd.read_csv('../outputs/predicted_values.csv')
#%%
differences = tree_pred_vals - poly_pred_vals
differences.std()
#%%
differences['UHI Index'].hist(bins = 20)
plt.show()
#%%
df_clean_factor_temp
#%%
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer
from scipy.stats import median_abs_deviation
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df_orig = pd.read_csv("../data/training_data_with_features_corr.csv")
print("Columns:", df_orig.columns.tolist())

# We'll assume the target column is "UHI Index"
target = "UHI Index"

# Drop non-numeric columns (e.g., datetime, day, etc.)
cols_to_drop = ["datetime", "time", "spatial_ref", "EVI", "day"]
df = df_orig.drop(columns=cols_to_drop, errors="ignore").copy()

# (Optional) If you want to use only a subset of features (for example, ['B01', 'B04','B05', 'B12']),
# uncomment the following lines:
cols_to_keep = ['B01', 'B04', 'B05', 'B12', target]
df = df[cols_to_keep].copy()

# Force conversion of all columns except target to numeric (non-convertible become NaN)
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------------------------------------------
# Outlier Removal Functions
# ------------------------------------------------------------
def remove_outliers_iqr(df, factor=3):
    """Custom function: Drop rows with any numeric feature out of [Q1 - factor*IQR, Q3 + factor*IQR]."""
    print("Inside remove_outliers_iqr, initial shape:", df.shape)
    df.reset_index(drop=True, inplace=True)
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_iqr_drop(df, factor=3):
    """Drop rows with any numeric feature out of [Q1 - factor*IQR, Q3 + factor*IQR]."""
    df_clean = df.copy().reset_index(drop=True)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_iqr_nan(df, factor=3):
    """Replace outliers with NaN using IQR method for numeric columns (except target)."""
    df_mod = df.copy()
    numeric_cols = df_mod.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]
    for col in numeric_cols:
        Q1 = df_mod[col].quantile(0.25)
        Q3 = df_mod[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_mod.loc[(df_mod[col] < lower_bound) | (df_mod[col] > upper_bound), col] = np.nan
    return df_mod

def remove_outliers_zscore_drop(df, threshold=4):
    """Drop rows with any numeric feature with robust z-score above threshold."""
    df_clean = df.copy().reset_index(drop=True)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]
    for col in numeric_cols:
        med = df_clean[col].median()
        mad = median_abs_deviation(df_clean[col], scale='normal')
        if mad == 0:
            continue
        z = (df_clean[col] - med) / mad
        df_clean = df_clean[(np.abs(z) <= threshold)]
    return df_clean

def remove_outliers_zscore_nan(df, threshold=3):
    """Replace values with NaN if their robust z-score exceeds threshold."""
    df_mod = df.copy()
    numeric_cols = df_mod.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]
    for col in numeric_cols:
        med = df_mod[col].median()
        mad = median_abs_deviation(df_mod[col], scale='normal')
        if mad == 0:
            continue
        z = (df_mod[col] - med) / mad
        df_mod.loc[np.abs(z) > threshold, col] = np.nan
    return df_mod

def winsorize_df(df, lower_quantile=0.05, upper_quantile=0.95):
    """Winsorize each numeric column (except target) by clipping at given quantiles."""
    df_mod = df.copy()
    numeric_cols = df_mod.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target]
    for col in numeric_cols:
        lower = df_mod[col].quantile(lower_quantile)
        upper = df_mod[col].quantile(upper_quantile)
        df_mod[col] = df_mod[col].clip(lower, upper)
    return df_mod

# ------------------------------------------------------------
# Define Imputation Functions (without MissForest)
# ------------------------------------------------------------
def impute_mean(df):
    return df.fillna(df.mean())

def impute_median(df):
    return df.fillna(df.median())

def impute_knn(df, n_neighbors=5):
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    if df.shape[0] == 0:
        return df
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df_imputed

def impute_iterative(df):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer(random_state=42)
    if df.shape[0] == 0:
        return df
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df_imputed

# ------------------------------------------------------------
# Iterate over combinations of outlier removal and imputation methods,
# then compute correlation matrix and report average absolute correlation
# between each feature and the target variable.
# ------------------------------------------------------------
outlier_methods = {
    "none": lambda df: df.copy(),
    "iqr": remove_outliers_iqr,          # the new function (prints df and resets index)
    "iqr_drop": remove_outliers_iqr_drop,
    "iqr_nan": remove_outliers_iqr_nan,
    "zscore_drop": remove_outliers_zscore_drop,
    "zscore_nan": remove_outliers_zscore_nan,
    "winsorize": winsorize_df
}

imputation_methods = {
    "none": lambda df: df.copy(),  # no imputation
    "mean": impute_mean,
    "median": impute_median,
    "knn": impute_knn,
    "iterative": impute_iterative
}

results = []

# We'll work only on the numeric DataFrame (including target)
df_numeric = df.copy()
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

for outlier_name, outlier_func in outlier_methods.items():
    df_out = outlier_func(df_numeric)
    for imp_name, imp_func in imputation_methods.items():
        df_imp = imp_func(df_out)
        # Skip if resulting DataFrame is empty
        if df_imp.shape[0] == 0:
            continue
        # Drop any remaining rows with missing target values
        df_imp = df_imp.dropna(subset=[target])
        if df_imp.shape[0] == 0:
            continue
        corr = df_imp.corr(method='spearman')
        if target in corr.columns:
            corr_target = corr[target].drop(target)
            avg_abs_corr = np.mean(np.abs(corr_target))
        else:
            avg_abs_corr = np.nan
        results.append({
            "outlier_method": outlier_name,
            "imputation_method": imp_name,
            "avg_abs_corr_with_target": avg_abs_corr,
            "n_samples": df_imp.shape[0]
        })

results_df = pd.DataFrame(results)
print("Comparison of methods:")
print(results_df.sort_values("avg_abs_corr_with_target", ascending=False))

# Optionally, plot the correlation matrix for the best combination:
best = results_df.sort_values("avg_abs_corr_with_target", ascending=False).iloc[0]
print("Best combination:", best)

# For example, using median imputation after IQR (with NaN replacement):
best_df = impute_median(remove_outliers_iqr_nan(df_numeric))
corr_best = best_df.corr(method='spearman')

plt.figure(figsize=(12,10))
sns.heatmap(corr_best, cmap=sns.diverging_palette(220, 20, as_cmap=True), annot=True, fmt=".2f", center=0)
plt.title("Correlation Matrix with Best Outlier/Imputation Method")
plt.show()

#%%
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assume df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build a Pipeline
# ------------------------------
# The pipeline includes optional preprocessing steps and then a model.
# By default, these steps are set to "passthrough" so that GridSearchCV can decide.
pipeline = Pipeline([
    ("variance_threshold", "passthrough"),
    ("select_from_model", "passthrough"),
    ("select_kbest", "passthrough"),
    ("scaler", StandardScaler()),
    ("model", ExtraTreesRegressor(random_state=42))
])

# ------------------------------
# 3. Define Parameter Grid for Multiple Models
# ------------------------------
param_grid = [
    {   # ExtraTreesRegressor settings
        "variance_threshold": [VarianceThreshold(threshold=0.001), VarianceThreshold(threshold=0.01), "passthrough"],
        "select_from_model": [
            SelectFromModel(estimator=ExtraTreesRegressor(n_estimators=300, random_state=42), threshold="mean"),
            SelectFromModel(estimator=ExtraTreesRegressor(n_estimators=300, random_state=42), threshold="0.99*mean"),
            "passthrough"
        ],
        "select_kbest": [SelectKBest(score_func=f_regression, k=5), "passthrough"],
        "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler(), QuantileTransformer(n_quantiles=100, output_distribution='normal'), "passthrough"],
        "model": [ExtraTreesRegressor(random_state=42)],
        "model__n_estimators": [300, 600, 900],
        "model__max_depth": [None, 50, 100],
        "model__min_samples_split": [2, 3, 4],
        "model__max_features": ["sqrt", "log2", None]
    },
    {   # RandomForestRegressor settings
        "variance_threshold": [VarianceThreshold(threshold=0.01), "passthrough"],
        "select_from_model": [
            SelectFromModel(estimator=RandomForestRegressor(n_estimators=300, random_state=42), threshold="mean"),
            "passthrough"
        ],
        "select_kbest": [SelectKBest(score_func=f_regression, k=5), "passthrough"],
        "scaler": [StandardScaler(), RobustScaler()],
        "model": [RandomForestRegressor(random_state=42)],
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 50, 100],
        "model__min_samples_split": [2, 3],
        "model__max_features": ["sqrt", "log2", None]
    },
    {   # GradientBoostingRegressor settings
        "variance_threshold": ["passthrough"],
        "select_from_model": ["passthrough"],
        "select_kbest": [SelectKBest(score_func=f_regression, k=5), "passthrough"],
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "model": [GradientBoostingRegressor(random_state=42)],
        "model__n_estimators": [300, 600],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 3]
    },
    {   # XGBRegressor settings (XGBoost)
        "variance_threshold": ["passthrough"],
        "select_from_model": ["passthrough"],
        "select_kbest": ["passthrough"],
        "scaler": [StandardScaler(), RobustScaler(), "passthrough"],
        "model": [XGBRegressor(random_state=42, objective='reg:squarederror')],
        "model__n_estimators": [300, 600],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1],
        "model__subsample": [0.8, 1.0]
    }
]

# ------------------------------
# 4. Run GridSearchCV with Verbose Output
# ------------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=3  # Verbose to get real-time output
)

print("Running GridSearchCV on the robust pipeline...")
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("Best CV R²: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 5. Evaluate on Test Set
# ------------------------------
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# ------------------------------
# 6. Refit on Entire Dataset and Save the Pipeline
# ------------------------------
best_pipeline.fit(X, y)
model_filename = "trained_model_pipeline_robust_multi.pkl"
joblib.dump(best_pipeline, model_filename)
print("Pipeline saved as:", model_filename)

#%%
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assume df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build the Pipeline
# ------------------------------
# The pipeline uses a scaler and then the XGBRegressor.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(random_state=42, objective='reg:squarederror'))
])

# ------------------------------
# 3. Define a Refined Hyperparameter Grid for XGBRegressor
# ------------------------------
param_grid = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [3, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0],
    "model__gamma": [0.1, 0.3, 0.5],
    "model__reg_alpha": [0.01, 0.1],
    "model__reg_lambda": [1, 2],
    "model__min_child_weight": [1, 3]
}

# ------------------------------
# 4. Run GridSearchCV with Verbose Output
# ------------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=3  # This level prints detailed progress information in real-time.
)

print("Running GridSearchCV for XGBRegressor with refined hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("Best CV R²: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 5. Evaluate on Test Set
# ------------------------------
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# ------------------------------
# 6. Refit on Entire Dataset and Save the Pipeline
# ------------------------------
best_pipeline.fit(X, y)
model_filename = "trained_xgb_pipeline_refined.pkl"
joblib.dump(best_pipeline, model_filename)
print("Pipeline saved as:", model_filename)

#%%
import pandas as pd
import numpy as np
import joblib
import sklearn
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Disable metadata routing to avoid tag issues
sklearn.set_config(enable_metadata_routing=False)

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assume df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build the Pipeline
# ------------------------------
# The pipeline now includes:
#   - SelectFromModel: to select features based on importance.
#   - SimpleImputer: to fill missing values (median strategy).
#   - Scaler: RobustScaler.
#   - Model: XGBRegressor with fixed best hyperparameters.
pipeline = Pipeline([
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.99*mean"
    )),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
    ("model", XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1,
        min_child_weight=1
    ))
])

# ------------------------------
# 3. Define a Refined Hyperparameter Grid for XGBRegressor
# ------------------------------
param_grid = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [3, 5, 7, 10],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.1, 0.3, 0.5],
    "model__reg_alpha": [0, 0.01, 0.1],
    "model__reg_lambda": [1, 1.5, 2],
    "model__min_child_weight": [1, 3, 5]
}

# ------------------------------
# 4. Run GridSearchCV with Verbose Output and Error Score Handling
# ------------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=3,
    error_score=-np.inf  # Penalize configurations that fail (or yield constant predictions)
)

print("Running GridSearchCV for XGBRegressor with refined hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("Best CV R²: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 5. Evaluate on Test Set
# ------------------------------
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# ------------------------------
# 6. Refit on Entire Dataset and Save the Pipeline
# ------------------------------
best_pipeline.fit(X, y)
model_filename = "trained_xgb_pipeline_refined.pkl"
joblib.dump(best_pipeline, model_filename)
print("Pipeline saved as:", model_filename)

#%%
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import randint, uniform

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assume df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build the Pipeline
# ------------------------------
pipeline = Pipeline([
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.99*mean"
    )),
    ("scaler", RobustScaler()),
    ("model", XGBRegressor(
        random_state=42,
        objective='reg:squarederror'
    ))
])

# ------------------------------
# 3. Define Hyperparameter Grid for Tuning
# ------------------------------
param_grid = {
    "model__n_estimators": randint(100, 1000),  # Number of trees
    "model__max_depth": randint(3, 50),       # Maximum depth of a tree
    "model__learning_rate": uniform(0.01, 0.3),  # Learning rate
    "model__subsample": uniform(0.6, 0.1),    # Subsample ratio of the training instances
    "model__colsample_bytree": uniform(0.6, 0.1),  # Subsample ratio of columns
    "model__gamma": uniform(0, 0.1),          # Minimum loss reduction to make a split
    "model__reg_alpha": uniform(0, 1),        # L1 regularization term
    "model__reg_lambda": uniform(0, 1),       # L2 regularization term
    "model__min_child_weight": randint(1, 10)  # Minimum sum of instance weight needed in a child
}

# ------------------------------
# 4. Perform Randomized Search for Hyperparameter Tuning
# ------------------------------
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=200,  # Number of parameter settings to sample
    scoring="neg_mean_absolute_percentage_error",  # Metric to optimize
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Print progress
    random_state=42,
    n_jobs=-1  # Use all available cores
)

print("Performing randomized search for hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found:", random_search.best_params_)
print("Best cross-validation score (MAPE):", -random_search.best_score_)

# ------------------------------
# 5. Evaluate the Best Model on the Test Set
# ------------------------------
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# ------------------------------
# 6. Refit on Entire Dataset and Save the Pipeline
# ------------------------------
best_model.fit(X, y)
model_filename = "trained_xgb_pipeline_tuned.pkl"
joblib.dump(best_model, model_filename)
print("Tuned pipeline saved as:", model_filename)
#%%
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import randint, uniform

# ------------------------------
# 1. Load Data and Preprocessing
# ------------------------------
# Assume df_clean_factor3 is your preprocessed DataFrame
data = df_clean_factor3.copy()

# Drop irrelevant columns
cols_to_drop = ["latitude", "longitude", "datetime", "distance", "distance_meters", "time", "spatial_ref", "EVI"]
data = data.drop(columns=cols_to_drop, errors="ignore")

# Define target variable
target = "UHI Index"
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets (25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# 2. Build the Pipeline
# ------------------------------
pipeline = Pipeline([
    ("select_from_model", SelectFromModel(
        estimator=ExtraTreesRegressor(n_estimators=300, random_state=42),
        threshold="0.99*mean"
    )),
    ("scaler", RobustScaler()),
    ("model", XGBRegressor(
        random_state=42,
        objective='reg:squarederror'
    ))
])

# ------------------------------
# 3. Define Hyperparameter Grid for Tuning
# ------------------------------
param_grid = {
    "model__n_estimators": randint(100, 500),  # Number of trees
    "model__max_depth": randint(3, 10),       # Maximum depth of a tree
    "model__learning_rate": uniform(0.01, 0.3),  # Learning rate
    "model__subsample": uniform(0.6, 0.4),    # Subsample ratio of the training instances
    "model__colsample_bytree": uniform(0.6, 0.4),  # Subsample ratio of columns
    "model__gamma": uniform(0, 0.5),          # Minimum loss reduction to make a split
    "model__reg_alpha": uniform(0, 1),        # L1 regularization term
    "model__reg_lambda": uniform(0, 1),       # L2 regularization term
    "model__min_child_weight": randint(1, 10)  # Minimum sum of instance weight needed in a child
}

# ------------------------------
# 4. Perform Randomized Search for Hyperparameter Tuning
# ------------------------------
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings to sample
    scoring="neg_mean_absolute_percentage_error",  # Metric to optimize
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Print progress
    random_state=42,
    n_jobs=-1  # Use all available cores
)

print("Performing randomized search for hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found:", random_search.best_params_)
print("Best cross-validation score (MAPE):", -random_search.best_score_)

# ------------------------------
# 5. Evaluate the Best Model on the Test Set
# ------------------------------
best_model = random_search.best_estimator_

# Extract the final estimator (XGBRegressor) from the pipeline
final_estimator = best_model.named_steps['model']

# Transform the test data using the pipeline steps before the final estimator
X_trans = best_model.named_steps['scaler'].transform(
    best_model.named_steps['select_from_model'].transform(X_test)
)

# Predict using the final estimator
y_pred = final_estimator.predict(X_trans)

# Evaluate the predictions
print("Test R²: {:.4f}".format(r2_score(y_test, y_pred)))
print("Test MAPE: {:.2f}%".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# ------------------------------
# 6. Refit on Entire Dataset and Save the Pipeline
# ------------------------------
best_model.fit(X, y)
model_filename = "trained_xgb_pipeline_tuned.pkl"
joblib.dump(best_model, model_filename)
print("Tuned pipeline saved as:", model_filename)
#%%
