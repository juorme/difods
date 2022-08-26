import pyodbc
import pandas as pd



conn = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'EG_BD',
                      UID = 'ussifods',
                      PWD = 'sifods')

df = pd.read_sql_query("select deser.*, cur.cursos_matriculados from  dbo.desercion2022 deser left join ind_cursos_docente_2022 cur on deser.DNI_CARACTERIZADO=cur.IDNUMBER where deser.CURID IN (280,289,298)",conn)
df.shape



# Variable tipo de insstitucion 
conn1 = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = 'med000008646',
                      DATABASE = 'db_sifods',
                      UID = 'ussifods',
                      PWD = 'sifods')

df1 = pd.read_sql_query("select  DNI, EDAD,RANGO_EDAD,D_COD_CAR from dct.[maestro.nexus_escale]",conn1)
df1.shape                      

# Datos de mesa de ayuda
df2 = pd.read_csv("D:/data/mesaayuda.csv", sep=";" )
df2.shape

#Datos rueba pun
df3 = pd.read_csv("D:/data/pruebaspun.csv",sep=";")
df3.DNI = df3['DNI'].astype('object')


m1 = pd.merge(df , df1 , how="left",left_on="DNI_CARACTERIZADO", right_on="DNI")
m2 = pd.merge(m1 , df2 , how="left",left_on="DNI_CARACTERIZADO",right_on="DNI")
m3 = pd.merge(m2 , df3 , how="left",left_on="DNI_x",right_on="DNI")



m3.to_csv("D:/variblesnuevas.csv",sep=";")