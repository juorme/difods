#Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
 

import warnings
warnings.filterwarnings('ignore')



# 1. Seleccion de datos
## Credenciales de autentificacion
# Conexión a base de datos
conn = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'BD_BI',
                      UID = 'ussifods',
                      PWD = 'sifods')

#Extraccion del dataset inical 
query1 = """select *  from [dbo].[TMP_ENCUESTAS]"""
df = pd.read_sql_query(query1,conn)
#Cerrar la conexion
conn.close()


# 2. Procesamiento
#Numero de docentes unicos y cursos unicos de la data inicial 
docentes_unicos = df.USUARIO.unique()
print(' # Numero de docentes unicos: ' + str(len(docentes_unicos)))
cursos_unicos= df.CURID.unique()
print(' # Numero de cursos unicos: ' + str(len(cursos_unicos)))

#Numero de docentes por Cursos 
df1 = df.groupby('DES_CURSO')['IDNUMBER'].nunique()
df1 = pd.DataFrame(df1)
df1.reset_index(inplace=True)
df1.sort_values(by ='IDNUMBER', ascending=False,inplace=True)
df1.head()

# 3. Transformacion
# Comprimir opción elegida
print(df['OPCION_ELEGIDA'].unique())
df.groupby('OPCION_ELEGIDA')['OPCION_ELEGIDA'].count()
## Reemplazamos los valores null por 3
df['OPCION_ELEGIDA']=df['OPCION_ELEGIDA'].fillna(3)
df['OPCION_ELEGIDA']=df['OPCION_ELEGIDA'].astype('int64')

#Curso mejor recomendado
mejor_puntuacion = df.groupby('CURID')['OPCION_ELEGIDA'].mean()
mejor_puntuacion = pd.DataFrame(mejor_puntuacion)
mejor_puntuacion.reset_index(inplace=True)
mejor_puntuacion.sort_values(by= 'OPCION_ELEGIDA', ascending=False, inplace= True)
mejor_puntuacion.head(5)

df0 = df.groupby(['CURID','IDNUMBER'])['OPCION_ELEGIDA'].mean()
df0 = pd.DataFrame(df0)
df0.reset_index(inplace=True)
df0.columns = ['CURID','IDNUMBER','PUNTUACION']
df0['PUNTUACION']= np.round(df0['PUNTUACION'],2)
df0.head(10)


# Pivot de la tabla inicial idnumber, curid y puntuacion
df_pivot = df0.pivot(index='IDNUMBER',columns='CURID',values='PUNTUACION')
df_pivot.head()


# 4. Mineria de datos 
# Calculando matrix sparsity 
sparsity_count = df_pivot.isnull().values.sum()
# Contar todas las celdas
full_cont = df_pivot.size
# Numero de escazes del dataset
sparsity = round(sparsity_count / full_cont, 3)
print('El dataset tiene una escasez de : ' + str(sparsity*100)+ '%')

# Contar las celdas ocupadas por columna
count_celdas_ocupadas = df_pivot.notnull().sum()
# Ordenar el resultado de mayot a menor
sorted_count_celdas_ocupadas = count_celdas_ocupadas.sort_values()
# Plot Histograma
sorted_count_celdas_ocupadas.hist()
plt.show()

# Descomposicion de valores singulares 
# Obtener la valoración media de cada usuario
avg_rating = df_pivot.mean(axis=1)
# Centrar las valoraciones de los usuarios entorno a 0
df_pivot_centered = df_pivot.sub(avg_rating, axis=0)
# Rellenar los datos que faltan con 0
df_pivot_centered.fillna(0, inplace=True)
# Comprobar la matriz centrada
print(df_pivot_centered.mean(axis=1))

#Descomponer la Matrix
U, sigma, Vt = svds(df_pivot_centered)
#Converitir el sigma en la diagonal de la matriz
sigma = np.diag(sigma)

#Producto de puntos
U_sigma = np.dot(U,sigma)
#Producto de puntos del resultado
U_sigma_Vt = np.dot(U_sigma, Vt)
#Las medias de las filas contenidas
Calificaciones_descentradas = U_sigma_Vt + avg_rating.values.reshape(-1,1)

calc_pred_df = pd.DataFrame(Calificaciones_descentradas,
                            index=df_pivot.index,
                            columns=df_pivot.columns)

print(calc_pred_df)

# 5. Interpretacion y Evaluacion 
# Comparar los metodos de recomendación
# Extraer los valores verdaderos para comparar las predicciones
actual_valor = df_pivot.iloc[:50, :15].values
predict_values = calc_pred_df.iloc[:50,:15].values 
mask = ~np.isnan(actual_valor)
print(mean_squared_error(actual_valor[mask], predict_values[mask],squared=False))

calc_pred_df = pd.DataFrame(calc_pred_df)
calc_pred_df=calc_pred_df.reset_index()
calc_pred_df.head()

# Obtener dataset de resultados
df_f = calc_pred_df.melt(id_vars='IDNUMBER',var_name='CURID',value_name='PUNTUACION')
df_f['PUNTUACION'] = np.round(df_f['PUNTUACION'],3)
df_f.head()

# Recomendacion de cursos por docente
# definir el docente
df_f[df_f['IDNUMBER']== '00124735'].sort_values(by='PUNTUACION', ascending= False).head(3)

# Carga de docentes 

# Credenciales de acceso  a base de datos BI
conn2 =  pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'BD_BI',
                      UID = 'ussifods',
                      PWD = 'sifods')

cursor = conn2.cursor()
# Borrar los datos de la tabla
cursor.execute("TRUNCATE TABLE [dbo].[TMP_RECOMENDACION]")
conn2.commit()

# Insertar valores 
sql_insert = """ INSERT INTO [dbo].[TMP_RECOMENDACION] VALUES (?,?,?) """
val = df_f[['IDNUMBER','CURID','PUNTUACION']].values.tolist()
cursor.executemany(sql_insert,val)
conn2.commit()
# Cerrar las conexiones
cursor.close()
conn2.close()

#Final
print("Ejecución exitosa : \n" + "se Cargó " + str(df_f.shape[0]) + " registros" )
