import pandas as pd
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('data/AREAS_DEFORESTADAS_CHOCO_20240608.csv')
print(data.head())

# Gráfico de barras - Área deforestada por año
plt.figure(figsize=(10, 6))
sns.barplot(x='AÑO', y='AREA_Ha', data=data, estimator=sum)
plt.title('Área Deforestada por Año')
plt.xlabel('Año')
plt.ylabel('Área Deforestada (Ha)')
plt.show()

# Gráfico de pastel - Proporción de deforestación por causa
causa_data = data.groupby('CAUSA')['AREA_Ha'].sum()
plt.figure(figsize=(8, 8))
causa_data.plot.pie(autopct='%1.1f%%')
plt.title('Proporción de Deforestación por Causa')
plt.ylabel('')
plt.show()


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


# Aplicar la función de conversión a las columnas de latitud y longitud
data['LATITUD'] = data['LATITUD'].apply(dms_to_decimal)
data['LONGITUD'] = data['LONGITUD'].apply(dms_to_decimal)

# Filtrar filas con coordenadas nulas (conversiones fallidas)
data = data.dropna(subset=['LATITUD', 'LONGITUD'])

# Parseo de los datos a un grafo
graph = nx.from_pandas_edgelist(data, source="MUNICIPIO", target="CAUSA", edge_attr="AREA_Ha")

# for idx, row in data[data['MUNICIPIO'] == "ACANDÍ"].iterrows():
#     graph.add_node(row['CAUSA'])
#     graph.add_node(row['MUNICIPIO'])
#     graph.add_edge(row['CAUSA'], row['MUNICIPIO'])

# pos = nx.spring_layout(graph)
# fig, ax = plt.subplots(figsize=(10, 8))
#
# nx.draw(graph, with_labels=True, ax=ax)


# Convertir a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.LONGITUD, data.LATITUD))
gdf = gdf.set_crs(epsg=4326)  # Establecer el CRS a WGS84
gdf = gdf.to_crs(epsg=3857)  # Convertir a Web Mercator para contextily

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 8))

# Añadir puntos al gráfico
gdf.plot(ax=ax, marker='o', color='red', markersize=5, alpha=0.6)

# Añadir mapa base
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Títulos y etiquetas
plt.title('Distribución Geográfica de la Deforestación')
plt.xlabel('Longitud')
plt.ylabel('Latitud')

plt.show()
