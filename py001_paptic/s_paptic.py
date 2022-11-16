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
from sklearn.cluster import KMeans
from scipy import stats
import pyodbc


## 01 Seleccion 
# Conexión a base de datos
conn = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'db_sifods',
                      UID = 'ussifods',
                      PWD = 'sifods')

# Tabla de las notas en los cursos de acompañatic
#El siguiente query hace un cruce con la tabla transaccional.oferta_formativa_agrupamiento para obtener el curid de los cursos que estan habilitados
#el cruce siguiente se realiza con la vista vw_sistema.mooc_carga_cumplimiento_pap para obtener las notas de los docentes que se encuentran con matricula vigente
#Ademas a ello se tiene que verificar que la nota del docente este homolñogado con el status de la tabla sistema.mooc_carga_cumplimiento 
query1 = """SELECT act.CANAL,act.CAMPUS,act.CURID,act.FULLNAME,act.IDNUMBER,act.NOMBRE_ACTIVIDAD,act.FECHA_REALIZADA,act.FINALGRADE
                FROM [acfm].[sistema.mooc_carga_actividad] act
                LEFT JOIN (SELECT CURID,IDNUMBER,FORMAT(FECHA_CULMINARON,'yyyy-MM-dd') AS FECHA1 FROM [acfm].[vw_sistema.mooc_carga_cumplimiento_pap]) cur on act.IDNUMBER =cur.IDNUMBER and act.CURID=cur.CURID and FORMAT(act.FECHA_REALIZADA,'yyyy-MM-dd')=cur.FECHA1
                WHERE act.CURID IN (SELECT CURID FROM [acfm].[transaccional.oferta_formativa_agrupamiento] WHERE CAMPUS = 7) 
                                AND act.CAMPUS=7 
                                AND act.NOMBRE_ACTIVIDAD IN ('Cuestionario de entrada','Cuestionario de salida')
                                AND act.PROGRESO IN ('COMPLETADO','COMPLETADO APROBADO','COMPLETADO DESAPROBADO')"""


# Tabla en donde se obtiene las variables adicionales de los docentes 
query2 = """SELECT DNI,NIVEL_NEXUS,DESCRIPCION_CARGO,SITUACION_LABORAL,EDAD,RANGO_EDAD,D_DPTO,D_PROV,D_DIST,D_DREUGEL,ESCALA_DIFODS,DAREACENSO FROM dct.[maestro.nexus_escale]"""


# Extracción de datos desde las tablas mencionadas 
df = pd.read_sql_query(query1,conn)
docentes = pd.read_sql_query(query2,conn)

# Datos iniciales 
print("Data Set inicial de cumplimiento : " + str(df.shape))
print("Data Set inicial de caracterizacion de docentes" + str(docentes.shape))

# Cerrar la conexión a la base de datos db_sifods
conn.close()


## 02 Procesamiento de datos 
# Transformación de caracteres en el dni 
df['IDNUMBER']=pd.to_numeric(df['IDNUMBER'], errors='coerce') 

# Elimina los valores nulos 
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
#reemplazar los valores null
dataf_1=dataf_1.dropna()
dataf_1['ESCALA_DIFODS']=dataf_1['ESCALA_DIFODS'].fillna(0)
 # print(dataf_1["NIVEL_NEXUS"].value_counts())


# Reclasificion variables
dataf_1['NIVEL_NEXUS'] = np.where(dataf_1['NIVEL_NEXUS'] =='Inicial - Jardín',1, np.where(dataf_1['NIVEL_NEXUS'] =='Primaria',2,np.where(dataf_1['NIVEL_NEXUS'] =='Secundaria',3,4)))
dataf_1['DAREACENSO'] = np.where(dataf_1['DAREACENSO'] == 'Rural', 1 , 2)

print('Numero Total de Registros de la Data Final : ' + str(dataf_1.shape))
dataf_1.head(3)


## 04 Mineria de datos 
#función para buscar el optimó numero de cluster(codo de jambu)
def grafico_codo(data_scaled):
  range_n_clusters = range(1, 11)
  inertias = []

  for n_clusters in range_n_clusters:
      modelo_kmeans = KMeans(n_clusters = n_clusters, n_init = 20, random_state = 123)
      modelo_kmeans.fit(X=data_scaled)
      inertias.append(modelo_kmeans.inertia_)

  fig, ax = plt.subplots(1, 1)
  ax.plot(range_n_clusters, inertias, marker='o')
  ax.set_title("Número Óptimo de Cluster")
  ax.set_xlabel('Número clusters')
  plt.show()



## 05 Evaluacion e interpretación
#datos
q1=tdf_pivot

# seleccion de variables
variables_2 = ['Cuestionario de entrada','Cuestionario de salida']

#Crear grafico 
#grafico_codo(q1.loc[:,variables_2])

# Reescalar las variables categoricas a numericas 
data_norm = StandardScaler().fit_transform(q1.loc[:,variables_2])

# los resultados del modelo se guardan en labels_ dentro del modelo mod_clus01
predicted_clusters_kmeans = KMeans(n_clusters=3, random_state=1).fit_predict(X=data_norm)

# indicadores de validacion davies_boulding y solhoute 
davies_bouldin_score_kmeans = davies_bouldin_score(data_norm, predicted_clusters_kmeans)
silhouette_score_kmeans = silhouette_score(data_norm, predicted_clusters_kmeans)
result_comp_clusters = pd.DataFrame(index=['Índice de Davies-Bouldin', 'Índice de Silhouette'])
result_comp_clusters['K-means'] = [davies_bouldin_score_kmeans, silhouette_score_kmeans]
print(result_comp_clusters)



# Agregar la columna de agrupacion (clusteres)
q1['clusters'] = predicted_clusters_kmeans + 1

# visulaizacion del dataframe con los resultados
q1.head(5)


## Carga de datos con los resultados del modelo  
conn3 = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'EG_BD',
                      UID = 'ussifods',
                      PWD = 'sifods')

cursor = conn3.cursor()
#Borrar los datos de la tabla ml.cluste_2022
cursor.execute("TRUNCATE TABLE ml.cluster_2022")
conn3.commit()

#Insertar valores 
sql_insert = """INSERT INTO [EG_BD].ml.cluster_2022 VALUES (?,?,?,?,?)"""
val = q1[['IDNUMBER','CURID','Cuestionario de entrada','Cuestionario de salida','clusters']].values.tolist()
cursor.executemany(sql_insert,val)
conn3.commit()
#Cerrar las conexiones
cursor.close()
conn3.close()

#Final
print("Ejecución exitosa: \n" + "Se cargaron " + str(q1.shape[0]) + " filas en la tabla")