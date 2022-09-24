#librerias
from asyncio.windows_events import NULL
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

query2 = """SELECT DNI,NIVEL_NEXUS,DESCRIPCION_CARGO,SITUACION_LABORAL,EDAD,RANGO_EDAD,D_DPTO,D_PROV,D_DIST,D_DREUGEL,ESCALA_DIFODS,DAREACENSO FROM [dct].[maestro.docente_integrado]"""


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

dataf_1.to_csv("D:/revisar.csv",sep=";")