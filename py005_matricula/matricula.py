import pandas as pd
import pyodbc

# Credenciales para la conexion
conn = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'db_sifods',
                      UID = 'ussifods',
                      PWD = 'sifods')

# Extracción de datos 
query1 = """SELECT 
DISTINCT(ID_OFERTA_FORMATIVA),a.NOMBRE, COUNT(DISTINCT(b.USUARIO_DOCUMENTO)) as DOCENTE
FROM [acfm].[transaccional.oferta_formativa_participante] b
INNER JOIN  [acfm].[maestro.oferta_formativa] a  on a.ID=b.ID_OFERTA_FORMATIVA
GROUP BY b.ID_OFERTA_FORMATIVA, a.NOMBRE"""

query2 = """SELECT DISTINCT(CURID), FULLNAME, COUNT(DISTINCT(IDNUMBER)) AS DOCENTE
FROM [acfm].[sistema.mooc_carga_data]
GROUP BY CURID, FULLNAME"""


df1 = pd.read_sql_query(query1,conn)
df2 = pd.read_sql_query(query2,conn)

# Cerrar la conexión a la base de datos db_sifods
conn.close()

# Limpieza de datos
df1["NOMBRE"] = df1['NOMBRE'].str.upper()
df2["FULLNAME"] = df2['FULLNAME'].str.upper()


# Unir datos 
merge_1 = pd.merge(df1,df2, left_on="NOMBRE",right_on="FULLNAME",how="inner")


def comparar_docentes(df):
    for i in range(len(df)):
        if df.iloc[i]["DOCENTE_x"] > df.iloc[i]["DOCENTE_y"]:
            return 1
        elif df.iloc[i]["DOCENTE_x"] < df.iloc[i]["DOCENTE_y"]:
            return 2 
        else: 
            df.iloc[i]["DOCENTE_x"] = df.iloc[i]["DOCENTE_y"]
            return 3

# Comparar en numero de docentes en el dataframe de unión 
merge_1["VAL_1"] =comparar_docentes(merge_1)

merge_1.rename(columns={'ID_OFERTA_FORMATIVA':'SIFODS_ID',
                            'NOMBRE' : 'SIFODS_NOMBRE',
                            'FULLNAME' : 'CAMPUS_NOMBRE',
                            'DOCENTE_x':'SIFODS_DOCENTE',
                            'DOCENTE_y':'CAMPUS_DOCENTE'}, inplace=True)

# Reporte General por curso 
reporte_1 = merge_1[merge_1['VAL_1']==1]
print(str(len(reporte_1))+ ' Cursos con mayor numero de docentes en SIFODS')
print(reporte_1)

reporte_2 = merge_1[merge_1['VAL_1']==2]
print(str(len(reporte_2))+ ' Cursos con mayor numero de docentes en CAMPUS_SIFODS')
print(reporte_2)

reporte_3 = merge_1[merge_1['VAL_1']==3]
print(str(len(reporte_3))+ ' Cursos con igual numero de docentes matriculados')
print(reporte_3)