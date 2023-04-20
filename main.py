import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import random
from sklearn.metrics import silhouette_score
#OMP_NUM_THREADS=4

random.seed(1234)

main_path = r"C:\Users\Alan Castaneda\Documents\GitHub\Microplasticos-FTIR-ML\\"

# IMPORTACION DE LOS DATOS
path_ = os.path.join(main_path, "muestras\\")

# -----------------------------------------------------------------------------
# Se renombran los archivos muestra para homogeneizar la información (se hace solo una vez)
# muestras = (os.listdir(path_))

# for i, file in enumerate(muestras):
#     new_file_name = "muestra{}.dpt".format(i)
#     os.rename(os.path.join(path_, file), os.path.join(path_, new_file_name))

# Se importa una muestra al azar y hacemos un .head() para corroborar la informacion
data1 = pd.read_csv(os.path.join(path_,"muestra1.dpt"), sep = "\t", header = None, dtype=np.float32)
data1.head(10)

# -----------------------------------------------------------------------------

# ARMADO DEL DATAFRAME
# Se rescata el numero de onda para armar el DataFrame añadiendolo a otro
numero_onda = data1.iloc[:, 0].tolist()
data = pd.DataFrame(data={"Numero de onda": numero_onda})

# Añadimos todas las muestras en el nuevo DataFrame
tempList = os.listdir(path_)

for x in range(0, len(tempList)):
    muestra = pd.read_csv(path_ + "muestra{}.dpt".format(x), sep = "\t", header = None)
    muestra = muestra.iloc[:, 1].tolist()
    
    name = "Muestra {}".format(x)
    df = pd.DataFrame(data = {name: muestra}) 
    data = pd.concat([data, df], axis=1)

# Generamos un archivo csv para visualizarlo a parte (opcional)
# data.to_csv('recopilac.csv', encoding='utf-8', index=False)    

# Corregimos el index en el nuevo DataFrame
data.index = data["Numero de onda"]
data = data.iloc[:, 1:]

# Funcion para visualizar una muestra en específico    
def PlotSample(n, data):
    A = data.iloc[:, n]
    
    plt.xlim(4000, 390)
    plt.ylim(0, 1.2)
    
    name = "Muestra {}".format(n)
    
    plt.title(name)
    plt.xlabel("Wavelength")
    plt.ylabel("Transmitance %")
    
    plt.plot(A)
    plt.show()

# Funcion para visualizar todas las muestras del DataFrame
def PlotAll(df):
   
    plt.xlim(4000, 390)
    plt.ylim(0, 1.2)
    
    plt.title("Distribution")
    plt.xlabel("Wavelength")
    plt.ylabel("Transmitance %")
    
    plt.plot(df, linewidth = 0.5)
    plt.show()
    
# Funcion para visualizar n cantidad de muestras en un plano
def PlotSamples(data, samples, clusteringName, numClase):
    plt.xlim(4000, 390)
    plt.ylim(0, 1.2)

    plt.title("Microplastic type {} of {} samples".format(numClase, clusteringName))
    plt.xlabel("Wavelength")
    plt.ylabel("Transmitance %")

    for n in samples:
        A = data.iloc[:, n]
        plt.plot(A, linewidth = 0.5)
    
    plt.show()

    
#-----------------------------------------------------------------------------

# ELIMINACION DE RUIDO
# Eliminacion de los outliers para evitar ruido en el procesamiento; se observó 
# la gráfica y se notó que hay datos en una transmitancia menor a 0.8 y mayor a 
# 1, modificar de ser necesario

fig = plt.figure(dpi = 500)
PlotAll(data)
fig.savefig("DistribucionPreFiltrado")

outliers_list = []

for n in range(0, data.shape[1]):
    if (data.iloc[0, n] < 0.8):
        outliers_list.append(n)
        
# for i in range(0, data.shape[1]):
#     for j in range(0, data.shape[0]):
#         if (data.iloc[j, i] > 1.1):
#             outliers_list.append(i)

set_outliers = set(outliers_list)
outliers_list = (list(set_outliers))

# Opcional para ver la lista de muestras eliminadas
#print(outliers_list)

data.drop(data.columns[outliers_list], axis=1, inplace=True)

#-----------------------------------------------------------------------------

# REARMADO DEL DATASET

tamano_df = (data.shape)[1]

for i in range(0, tamano_df):    
    name = "Muestra {}".format(i)
    data.columns.values[i] = name
    i += 1
    
data.columns.values

# Eliminamos todas las variables temporales creadas para liberar memoria
del data1, df, i, muestra, n, name, numero_onda, outliers_list, path_,\
tamano_df, tempList, x, set_outliers

# Hacemos plot de todas las muestras para visualizarlas en un solo plot y corro-
# borar que se tenga una distribucion mas uniforme

fig = plt.figure(dpi=500)
PlotAll(data)
fig.savefig("DistribucionPostFiltrado")


#-----------------------------------------------------------------------------

# IMPORTACION DE MUESTRAS CONOCIDAS 

# Ruta de los datos
val_path = os.path.join(main_path, "blancos//")

# Lista de los nombres
lista_blancos = os.listdir(val_path)

# Creacion de un DataFrame nuevo de los datos de validacion
blanco = pd.read_csv(os.path.join(val_path, lista_blancos[0]), sep = "\t", header = None)
numero_onda = blanco.iloc[:, 0].tolist()
data_blancos = pd.DataFrame(data={"Numero de onda": numero_onda})

# Se importan los tipos de plasticos conocidos
for i in range(0, len(lista_blancos)):
    muestra = pd.read_csv(os.path.join(val_path, "{}".format(lista_blancos[i])), 
                          sep = "\t", header = None)
    muestra = muestra.iloc[:, 1].tolist()
    nombre = "{}".format(lista_blancos[i])
    dff = pd.DataFrame(data = {nombre: muestra})
    data_blancos = pd.concat([data_blancos, dff], axis=1)

# Se asigna el numero de onda al index
index_global = data_blancos["Numero de onda"]
data_blancos.index = index_global
data_blancos = data_blancos.iloc[:, 1:]

# Se añaden las muestras conocidas al dataset

data.index = index_global
data = pd.concat([data, data_blancos], axis=1)

#-----------------------------------------------------------------------------

# APLICACION DE PCA Y NORMALIZACION

# Para el algoritmo de PCA
dataset = data.T

# Estandarizar los datos
scaler = StandardScaler()
dataset_escalado = scaler.fit_transform(dataset)

# Aplicar PCA a los datos
pca = PCA()
pca.fit(dataset_escalado)

# Calcular el ratio de covarianza para cada componente principal
varianza_explicada = pca.explained_variance_ratio_

# Creacion de plot
plt.plot(range(1, len(varianza_explicada)+1), varianza_explicada, marker='o')
plt.title("Principal Component Analysis (PCA)")
plt.xlim(0, 12)
plt.ylim(0, 0.6)
plt.xlabel('Number of components')
plt.ylabel('Explained covariance ratio')
plt.savefig("PCA", dpi=500)
plt.show()

# Determinar el numero de componentes (en este caso el mejor numero 3)
# examinando la grafica y detectandolos en el codo de la funcion
pca = PCA(n_components = 3)
pca.fit(dataset_escalado)

# Aplicamos la transformacion de los datos
df = pca.fit_transform(dataset_escalado)

#-----------------------------------------------------------------------------

# APLICACION DE LOS ALGORITMOS

#------------------------------------------------------
# Parametros
numeroClusters = 12

# KMeans
kmeans = KMeans(n_clusters=numeroClusters)
kmeans.fit_predict(df)
labels_kmeans = kmeans.labels_

# Spectral Clustering
spectral = SpectralClustering(n_clusters=numeroClusters, eigen_solver='arpack', 
                              affinity="nearest_neighbors")
spectral.fit_predict(df)
labels_spectral = spectral.labels_

# Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=numeroClusters)
agglomerative.fit_predict(df)
labels_agglo = agglomerative.labels_

lista_algo = [labels_kmeans, labels_spectral, labels_agglo]

# Añadir los resultados del cluster al dataset
dataset_predicciones = dataset
dataset_predicciones_kmeans = dataset_predicciones
dataset_predicciones_spectral = dataset_predicciones
dataset_predicciones_agglo = dataset_predicciones


# Añadimos las predicciones a 3 nuevos dataframes, que despues recorreremos con un
# ciclo "for", para detectar las predicciones y hacer los plots de distribución de cada
# predicción en función de la muestra que es

# Para K-Means
dataset_predicciones_kmeans = dataset_predicciones_kmeans
dataset_predicciones_kmeans['KMeans Clustering'] = labels_kmeans
dataset_predicciones_kmeans = dataset_predicciones_kmeans.T
listaTemp = []
titulo = 'KMeans Clustering'
for j in range(0, numeroClusters):
    for i in range(0, dataset_predicciones_kmeans.shape[1]):
        if (dataset_predicciones_kmeans.iloc[-1, i] == j):
            listaTemp.append(i)
    fig = plt.figure(dpi = 500)
    PlotSamples(data, listaTemp, titulo, j)
    fig.savefig("{} clase {}".format(titulo, j))
    listaTemp.clear()   

# Para Spectral Clustering
dataset_predicciones_spectral = dataset_predicciones_spectral
dataset_predicciones_spectral['Spectral Clustering'] = labels_spectral
dataset_predicciones_spectral = dataset_predicciones_spectral.T
listaTemp = []
titulo = 'Spectral Clustering'
for j in range(0, numeroClusters):
    for i in range(0, dataset_predicciones_spectral.shape[1]):
        if (dataset_predicciones_spectral.iloc[-1, i] == j):
            listaTemp.append(i)
    fig = plt.figure(dpi = 500)
    PlotSamples(data, listaTemp, titulo, j)
    fig.savefig("{} clase {}".format(titulo, j))
    listaTemp.clear()  

# Para Agglomerative Clustering
dataset_predicciones_agglo = dataset_predicciones_agglo
dataset_predicciones_agglo['Agglomerative Clustering'] = labels_agglo
dataset_predicciones_agglo = dataset_predicciones_agglo.T
listaTemp = []
titulo = 'Agglomerative Clustering'
for j in range(0, numeroClusters):
    for i in range(0, dataset_predicciones_agglo.shape[1]):
        if (dataset_predicciones_agglo.iloc[-1, i] == j):
            listaTemp.append(i)    
    fig = plt.figure(dpi = 500)
    PlotSamples(data, listaTemp, titulo, j)
    fig.savefig("{} clase {}".format(titulo, j))
    listaTemp.clear()  
    

#-----------------------------------------------------------------------------

# VISUALIZACION

# Funcion para visualizar los dos subplots de los clusters
def ClustersVisualization(lbls, title):
    # Set de colores
    cmap = cm.get_cmap('Set1')
    colors = [cmap(lbl/numeroClusters) for lbl in lbls]
    
    # Visualizacion de resultados en 3D
    fig = plt.figure(figsize=(10, 5), dpi=500)
    fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(numeroClusters):
        ax1.scatter(df[lbls == i, 0], df[lbls == i, 1], 
                    df[lbls == i, 2], label=i, s=10, alpha=0.5)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend()
    # Elevacion y Azimuth para mover la vista
    ax1.view_init(elev=40, azim=50)
    
    # Visualizacion de resultados en 2D
    ax2 = fig.add_subplot(122)
    for i in range(numeroClusters):
        ax2.scatter(df[lbls == i, 0], df[lbls == i, 1], label=i, s=10, alpha=0.5)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    
    # Set the colors of the scatter plots
    ax1.scatter(df[:, 0], df[:, 1], df[:, 2], c=colors, s=10, alpha=0.5)
    ax2.scatter(df[:, 0], df[:, 1], c=colors, s=10, alpha=0.5)
    
    # Save the plot to a file with a custom name
    fig.savefig(title, dpi=500)
    plt.show()

# Graficacion de los diferentes clusters por separado en 3D y 2D
ClustersVisualization(labels_kmeans, "KMeans")
ClustersVisualization(labels_spectral, "Spectral")
ClustersVisualization(labels_agglo, "Agglomerative")

# Graficacion de los diferentes clusters en 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=500)
fig.suptitle('Clustering Results')
fig.subplots_adjust(top=0.75, wspace=0.3)

# K-Means
ax1.scatter(df[:, 0], df[:, 1], c=labels_kmeans)
ax1.set_title('K-Means')

# Spectral Clustering
ax2.scatter(df[:, 0], df[:, 1], c=labels_spectral)
ax2.set_title('Spectral Clustering')

# Agglomerative Clustering
ax3.scatter(df[:, 0], df[:, 1], c=labels_agglo)
ax3.set_title('Agglomerative Clustering')

fig.savefig("Clustering Results", dpi=500)
plt.show()


#-----------------------------------------------------------------------------

# EXPORTACION DE LOS RESULTADOS

# Exportacion de los datos a Excel
dataset.T.to_csv('Clasificaciones.csv', encoding='utf-8', index=True)

# Ruta en donde se guardaran las muestras clasificadas
output_dir = os.path.join(main_path, "SamplesPredictions\\")

# Iteracion sobre cada muestra en "dataset"
for i, row in dataset.iterrows():
    # Creacion de un nombre de archivo para el formato .txt
    filename = os.path.join(output_dir, f'{i}.txt')
    # Escritura de los datos
    row.to_csv(filename, sep='\t', index=True, header=True)