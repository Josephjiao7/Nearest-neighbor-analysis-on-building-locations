import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import geopandas as gpd

# Read the GeoPandas DataFrame from a shapefile
df = gpd.read_file('buildings.shp')

# Extract the 'id' column
id = df['id']

# Use the KDTree algorithm to calculate the proximity of buildings
tree = KDTree(df[['x', 'y']].values)
dist, ind = tree.query(df[['x', 'y']].values, k=2)
df['dist'] = dist[:, 1]
distance = df['dist']

# Concatenate 'id' and 'distance' columns to create the result DataFrame
result = pd.concat([id, distance], axis=1)

# Save the computation results to a file
result.to_csv('buildings_nearest.csv', index=False)
