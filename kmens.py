import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Lee los puntos desde un archivo CSV
data = pd.read_csv('C:/Users/Enrique/Desktop/kmens/datae.csv')

X = data.values  # Convierte los datos a un array de NumPy

# Especifica el nuevo número de clusters (k) y n_init
k = 3
n_init_value = 10

# Crea el modelo K-Means
kmeans = KMeans(n_clusters=k, n_init=n_init_value)
kmeans.fit(X)

# Obtiene las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Imprime las asignaciones de clusters para cada punto y la distancia a su centroide
for i in range(len(X)):
    dist = np.linalg.norm(X[i] - centroids[labels[i]])
    print(f'Punto {i+1} ({X[i][0]}, {X[i][1]}) está asignado al Cluster {labels[i] + 1} con una distancia de {dist} al centroide')

# Cambia la lista de colores a colores válidos
colors = ["g", "r", "b"]

# Visualiza los resultados
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
    plt.text(X[i][0], X[i][1], f'p{i+1}', fontsize=8, ha='right')
    plt.plot([X[i][0], centroids[labels[i], 0]], [X[i][1], centroids[labels[i], 1]], 'k--')

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10, color='b', label='Centroides')

# Agrega nombres a los centroides con un desplazamiento mayor para evitar la superposición
for i in range(k):
    plt.text(centroids[i, 0] + 0.2, centroids[i, 1] + 0.2, f'C{i+1}', fontsize=12, ha='right')

for cluster in range(k):
    cluster_points = X[labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster], label=f'Cluster {cluster + 1}')

plt.legend()

# Ajusta los valores de los ejes x e y
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 11, 1))

plt.show()

# Solicitar entrada para un nuevo punto desde la consola
new_x = float(input("Ingrese la coordenada x del nuevo punto: "))
new_y = float(input("Ingrese la coordenada y del nuevo punto: "))
new_point = np.array([[new_x, new_y]])

# Clasificar el nuevo punto en el modelo existente
new_label = kmeans.predict(new_point)

# Imprime a qué centroide se asignó el nuevo punto y la distancia a cada centroide
print(f'El nuevo punto ({new_x}, {new_y}) está asignado al Centroide C{new_label[0] + 1} porque es el más cercano con una distancia de {np.linalg.norm(new_point - centroids[new_label[0]])}')
for i in range(k):
    dist_to_centroid = np.linalg.norm(new_point - centroids[i])
    print(f'La distancia del nuevo punto al Centroide C{i+1} es {dist_to_centroid}')

# Generar una segunda gráfica con solo los centroides y el nuevo punto
plt.figure()

# Visualizar los centroides
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10, color='b', label='Centroides')

# Agrega nombres a los centroides con un desplazamiento mayor para evitar la superposición
for i in range(k):
    plt.text(centroids[i, 0] + 0.2, centroids[i, 1] + 0.2, f'C{i+1}', fontsize=12, ha='right')

# Agregar el nuevo punto y su clasificación
plt.scatter(new_point[:, 0], new_point[:, 1], c=colors[new_label[0]], marker='o', label='Nuevo Punto')

# Dibujar una línea punteada desde el nuevo punto hasta su centroide
plt.plot([new_point[0, 0], centroids[new_label[0], 0]], [new_point[0, 1], centroids[new_label[0], 1]], 'k--')

plt.legend()

# Ajusta los valores de los ejes x e y
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 11, 1))

# Crear una lista para almacenar los datos de la tabla
table_data = []

# Agregar encabezados a la lista
table_data.append(["Cluster", "Punto", "Coordenadas", "Centroide"])

# Agregar datos a la lista
for i in range(len(X)):
    cluster_idx = labels[i]
    point_coords = f'({X[i][0]}, {X[i][1]})'
    centroid_coords = f'({centroids[cluster_idx][0]}, {centroids[cluster_idx][1]})'
    table_data.append([cluster_idx + 1, f'P{i + 1}', point_coords, centroid_coords])

# Imprimir la tabla
print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

plt.show()
