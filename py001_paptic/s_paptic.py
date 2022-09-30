#librerias
from asyncio.windows_events import NULL
from socketserver import DatagramRequestHandler
from sqlite3 import Cursor
from unittest import result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score, davies_bouldin_score

#!pip install scikit-learn-extra
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering

#!pip install bioinfokit
from bioinfokit.analys import stats
from scipy import stats
from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
import pyodbc

## Cursos disponibles de acompañatic
conn = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'db_sifods',
                      UID = 'ussifods',
                      PWD = 'sifods')

query1 = """SELECT act.CANAL,act.CAMPUS,act.CURID,act.FULLNAME,act.IDNUMBER,act.NOMBRE_ACTIVIDAD,act.FECHA_REALIZADA,act.FINALGRADE
                FROM [acfm].[sistema.mooc_carga_actividad] act
                LEFT JOIN (SELECT CURID,IDNUMBER,FORMAT(FECHA_CULMINARON,'yyyy-MM-dd') AS FECHA1 FROM [acfm].[vw_sistema.mooc_carga_cumplimiento_pap]) cur on act.IDNUMBER =cur.IDNUMBER and act.CURID=cur.CURID and FORMAT(act.FECHA_REALIZADA,'yyyy-MM-dd')=cur.FECHA1
                WHERE act.CURID IN (SELECT CURID FROM [acfm].[transaccional.oferta_formativa_agrupamiento] WHERE CAMPUS = 7) 
                                AND act.CAMPUS=7 
                                AND act.NOMBRE_ACTIVIDAD IN ('Cuestionario de entrada','Cuestionario de salida')
                                AND act.PROGRESO IN ('COMPLETADO','COMPLETADO APROBADO','COMPLETADO DESAPROBADO')"""

query2 = """SELECT DNI,NIVEL_NEXUS,DESCRIPCION_CARGO,SITUACION_LABORAL,EDAD,RANGO_EDAD,D_DPTO,D_PROV,D_DIST,D_DREUGEL,ESCALA_DIFODS,DAREACENSO FROM dct.[maestro.nexus_escale]"""


df = pd.read_sql_query(query1,conn)
docentes = pd.read_sql_query(query2,conn)

print("Data Set inicial de cumplimiento : " + str(df.shape))
print("Data Set inicial de caracterizacion de docentes" + str(docentes.shape))


#Procesamiento de datos 
#Transformación de caracteres en el dni 
df['IDNUMBER']=pd.to_numeric(df['IDNUMBER'], errors='coerce') 

#Elimina los valores nulos 
df.dropna(subset=['IDNUMBER'], inplace=True)



tdf = df.loc[:,['CURID','IDNUMBER','NOMBRE_ACTIVIDAD','FINALGRADE']]
tdf_pivot = pd.pivot_table(tdf,index=['CURID','IDNUMBER'],columns='NOMBRE_ACTIVIDAD', values='FINALGRADE').reset_index()

# Reemplazando valores nulos
tdf_pivot["Cuestionario de entrada"] = tdf_pivot["Cuestionario de entrada"].fillna(0)
tdf_pivot["Cuestionario de salida"] = tdf_pivot["Cuestionario de salida"].fillna(0)
tdf_pivot.isnull().sum()


docentes['DNI']=pd.to_numeric(docentes['DNI'], errors='coerce')
dataf_1 = tdf_pivot.merge(docentes , how='left', left_on='IDNUMBER', right_on='DNI' )
dataf_1.isnull().sum()


print("Dni no encontrados : "+ str(dataf_1['DNI'].isnull().sum()))

## 03 Transformacion
dataf_1.isnull().sum()

#reemplazar los valores null
dataf_1=dataf_1.dropna()
dataf_1['ESCALA_DIFODS']=dataf_1['ESCALA_DIFODS'].fillna(0)


print(dataf_1["NIVEL_NEXUS"].value_counts())


# Reclasificando variables
dataf_1['NIVEL_NEXUS'] = np.where(dataf_1['NIVEL_NEXUS'] =='Inicial - Jardín',1, np.where(dataf_1['NIVEL_NEXUS'] =='Primaria',2,np.where(dataf_1['NIVEL_NEXUS'] =='Secundaria',3,4)))
dataf_1['DAREACENSO'] = np.where(dataf_1['DAREACENSO'] == 'Rural', 1 , 2)

print('Numero Total de Registros de la Data Final : ' + str(dataf_1.shape))
dataf_1.head(3)


## 04 Mineria de datos 
# función para buscar el optimó numero de codo
# def grafico_codo(data_scaled):
#   range_n_clusters = range(1, 11)
#   inertias = []

#   for n_clusters in range_n_clusters:
#       modelo_kmeans = KMeans(n_clusters = n_clusters, n_init = 20, random_state = 123)
#       modelo_kmeans.fit(X=data_scaled)
#       inertias.append(modelo_kmeans.inertia_)

#   fig, ax = plt.subplots(1, 1)
#   ax.plot(range_n_clusters, inertias, marker='o')
#   ax.set_title("Número Óptimo de Cluster")
#   ax.set_xlabel('Número clusters')
#   plt.show()

# # función para prueba de normalidad
# def prueba_normalidad(datos, data_scaled):

#   df_swt = pd.DataFrame(index=['Estadístico de la prueba', 'p-value'])

#   for i, k in enumerate(datos.keys()):
#     (swt, swp) = stats.normaltest(data_scaled[:,i])
#     df_swt[k] = [float("{:.4f}".format(swt)), float("{:.4f}".format(swp))]
    
#   return(df_swt)

#Función grafico de cluster dos variables
# def grafico_cluster_principal(datos, c1, c2):
#   y = datos['clusters']
#   plt.figure(figsize=(15, 15))
#   fig, ax = plt.subplots()
#   sc = ax.scatter(datos[c1], datos[c2],c=y)
#   ax.legend(*sc.legend_elements(), title='Grupos')
#   plt.xlabel(c1,size=14)
#   plt.ylabel(c2,size=14)
#   #plt.axis("equal")
#   plt.title('Clusterización K-means(k=4)', size=18)
#   plt.show()

# Función grafico de cluster tres variables
# def grafico_cluster_3d(datos,x,y,z):
#     y1 = datos['clusters']
#     fig = plt.figure(figsize=(10,8))
#     ax = fig.add_subplot(111,projection='3d')
#     ax.view_init(15, 40)
#     plt.xlabel(x,size= 14)
#     plt.ylabel(y,size= 14)
#     plt.title("Clusterización K-means(k=4)",size= 20)
#     sc = ax.scatter(datos[x],datos[y],datos[z],c=y1)
#     ax.legend(*sc.legend_elements(), title='Grupos')
#     plt.show()

# Funcion para calcular prueba de kruskall_wallis , prueba de diferencia de grupos
# def kruskal_wallis(datos):
#   df_kwt = pd.DataFrame(index=['Estadístico de la prueba', 'p-value'])
#   var1 = ['NIVEL_NEXUS','EDAD','ESCALA_DIFODS','DAREACENSO','Cuestionario de entrada','Cuestionario de salida']
#   for c in var1:
#     datos_grupo1_temp  = datos.loc[datos['clusters']==1][c].values
#     datos_grupo2_temp  = datos.loc[datos['clusters']==2][c].values
#     datos_grupo3_temp  = datos.loc[datos['clusters']==3][c].values
#     datos_grupo4_temp  = datos.loc[datos['clusters']==4][c].values

#     (stkw, pvkw) = stats.kruskal(datos_grupo1_temp, datos_grupo2_temp, datos_grupo3_temp, datos_grupo4_temp)

#     df_kwt[c] = [float("{:.4f}".format(stkw)), float("{:.4f}".format(pvkw))]
#   return(df_kwt)

# funcion para comparar el numero de cluster
def comparar_clusters(data_scaled):
   for i in range(3,7) :
    predicted_clusters_kmeans = KMeans(n_clusters=i, random_state=1).fit_predict(X=data_scaled)
    davies_bouldin_score_kmeans = davies_bouldin_score(data_scaled, predicted_clusters_kmeans)
    silhouette_score_kmeans = silhouette_score(data_scaled, predicted_clusters_kmeans)

#     predicted_clusters_miniba = MiniBatchKMeans(n_clusters=i, random_state=2).fit_predict(X=data_scaled)
#     davies_bouldin_score_miniba = davies_bouldin_score(data_scaled, predicted_clusters_miniba)
#     silhouette_score_miniba = silhouette_score(data_scaled, predicted_clusters_miniba)

    result_comp_clusters = pd.DataFrame(index=['Índice de Davies-Bouldin', 'Índice de Silhouette'])
    result_comp_clusters['K-means'] = [davies_bouldin_score_kmeans, silhouette_score_kmeans]
#     result_comp_clusters['MiniBatchKM'] = [davies_bouldin_score_miniba, silhouette_score_miniba]
    return(result_comp_clusters)

def analisis_resultados(datosini,data_norm):
    var2 = ['NIVEL_NEXUS','EDAD','ESCALA_DIFODS','DAREACENSO','Cuestionario de entrada','Cuestionario de salida']
    # los resultados del modelo se guardan en labels_ dentro del modelo mod_clus01
    predicted_clusters_kmeans = KMeans(n_clusters=4, random_state=1).fit_predict(X=data_norm)
    datosini['clusters'] = predicted_clusters_kmeans + 1
    gr1 = datosini.groupby('clusters').agg(N = ("clusters", 'count'))
    #gr2 = gr1.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    print(gr1)
    #datosini['clusters'].value_counts()

    # Prueba de normalidad de var2
    #prueba_normalidad( datosini.loc[:,var2],data_norm)

    # Prueba no parametrica de diferencia de grupos
    # kruskal_wallis(datosini)




# Seleccion de variables y normalización 
variables = ['IDNUMBER','NIVEL_NEXUS','EDAD','ESCALA_DIFODS','DAREACENSO','Cuestionario de entrada','Cuestionario de salida']
var_2 = ['IDNUMBER','CURID','Cuestionario de entrada','Cuestionario de salida']

# Reescalar las variables categoricas a numericas 
data_norm = StandardScaler().fit_transform(dataf_1.loc[:,variables].drop('IDNUMBER',axis=1))

# los resultados del modelo se guardan en labels_ dentro del modelo mod_clus01
predicted_clusters_kmeans = KMeans(n_clusters=4, random_state=1).fit_predict(X=data_norm)

# indicaores de davies_boulding y solhoute 
davies_bouldin_score_kmeans = davies_bouldin_score(data_norm, predicted_clusters_kmeans)
silhouette_score_kmeans = silhouette_score(data_norm, predicted_clusters_kmeans)
result_comp_clusters = pd.DataFrame(index=['Índice de Davies-Bouldin', 'Índice de Silhouette'])
result_comp_clusters['K-means'] = [davies_bouldin_score_kmeans, silhouette_score_kmeans]

# Agregar la columna de agrupacín (clusteres)
data_fin =dataf_1.loc[:,var_2]
data_fin['clusters'] = predicted_clusters_kmeans + 1

data_fin.head(5)


# Insertar los datos 
conn3 = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'EG_BD',
                      UID = 'ussifods',
                      PWD = 'sifods')

cursor = conn3.cursor()

sql_insert = """INSERT INTO [EG_BD].ml.cluster_2022 (IDNUMBER,CURID,CUEST_ENTRADA,CUEST_SALIDA,CLUSTER) VALUES (?,?,?,?,?)"""
val = data_fin[['IDNUMBER','CURID','Cuestionario de entrada','Cuestionario de salida','clusters']].values.tolist()


cursor.executemany(sql_insert,val)
conn3.commit()




