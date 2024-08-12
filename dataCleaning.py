import pandas as pd
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from geopy.distance import great_circle
from sklearn.neighbors import NearestNeighbors

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('data/AREAS_DEFORESTADAS_CHOCO_20240608.csv')
head = data.head()
print(data.head())

# Mapa de calor - Distribución geográfica de la deforestación

def dms_to_decimal(dms_str):
    """Convert DMS (degrees, minutes, seconds) to decimal format, handling N/S/E/W directions."""
    try:
        if pd.notnull(dms_str):
            dms_str = str(dms_str).strip()  # Convert to string and strip whitespace
            parts = dms_str.replace('°', '').replace("'", '').replace('"', '').replace(',', '.').split()

            # Verificación del número de partes y la dirección
            if len(parts) == 4:
                degrees = float(parts[0])
                minutes = float(parts[1]) / 60
                seconds = float(parts[2]) / 3600
                direction = parts[3]

                decimal = degrees + minutes + seconds
                if direction in ['S', 'W']:
                    decimal = -decimal
                return decimal
            else:
                raise ValueError(f"Formato de coordenadas inválido: {dms_str}")
        else:
            return None
    except Exception as e:
        print(f"Error en la conversión: {e}")
        return None


df = data.copy()
df = df[df['MUNICIPIO'] == "ACANDÍ"]
df.reset_index(drop=True, inplace=True)
df['LATITUD'].dropna(inplace=True)
df['LONGITUD'].dropna(inplace=True)
# Aplicar la función de conversión a las columnas de latitud y longitud

df['LATITUD'] = df['LATITUD'].apply(dms_to_decimal)
df['LONGITUD'] = df['LONGITUD'].apply(dms_to_decimal)

# Filtrar filas con coordenadas nulas (conversiones fallidas)
df = df.dropna(subset=['LATITUD', 'LONGITUD'])

print(df)