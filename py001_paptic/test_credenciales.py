import pandas as pd
import pyodbc



# Conexión a base de datos
conn = pyodbc.connect(DRIVER = '{ODBC Driver 17 for SQL Server}',
                      SERVER = '10.200.9.11',
                      DATABASE = 'db_sifods',
                      UID = 'user_sifods',
                      PWD = 'user_sifods')


query1 = """SELECT  * FROM [stage].[acfm_sistema_mooc_carga_actividad]"""             

# Extracción de datos desde las tablas mencionadas 
df = pd.read_sql_query(query1,conn)

# Cerrar la conexión a la base de datos db_sifods
conn.close()