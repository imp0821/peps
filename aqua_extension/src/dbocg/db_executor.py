from db_config import read_config
import psycopg2
import logging
import numpy as np
from multiprocessing import Pool

class AQUAException(Exception):
    pass

def DBTensorInfo(self):
    def __init__(self, table_name):
        self.table_name = table_name
        self.create_q = ""
        self.insert_q = "" 
        self.data = None

class DBExecutor:
    def __init__(self):
        config = read_config()

        self.__pg_connect_str = config["PostgreSQLConnectString"]  
        self.__max_query_time = int(config["MaxQueryTimeSeconds"]) * 1000
    
    def __get_pg_cursor(self):
        try:
            conn = psycopg2.connect(self.__pg_connect_str)
            return conn.cursor()
        except psycopg2.OperationalError as e:
            logging.error("[DBExecutor] Could not connect to PG database")
            raise AQUAException("Could not connect to PG database") from e

    def test_connection(self):
        with self.__get_pg_cursor() as cur:
            try:
                cur.execute("SELECT 1")
                cur.fetchall()
            except Exception as e:
                logging.error("[DBExecutor] Could not connect to PG database")
                raise AQUAException("Could not connect to the PostgreSQL database.") from e
        return True
    

    def write_command(self, command):
        with self.__get_pg_cursor() as c:
            c.execute(command)
            c.execute("commit")
            c.close()

    def write_with_data(self, command):
        with self.__get_pg_cursor() as c:
            c.execute(command[0], (command[1],))
            c.execute("commit")
            c.close()
    

    def write_tensor_parallel(self, table_mapping_tensors):
        create_queries = []
        insert_queries = []
        delete_quereis = []
        for name, val in table_mapping_tensors.items():
            create_q = f"CREATE TABLE IF NOT EXISTS {name} (val REAL[][]);"
            create_queries.append(create_q)
            
            drop_q = f"DROP TABLE IF EXISTS {name};"
            delete_quereis.append(drop_q)

            insert_q = f"INSERT INTO {name} (val) VALUES (%s);"
            insert_queries.append((insert_q, table_mapping_tensors[name]))
        
        with self.__get_pg_cursor() as c:
            for q in create_queries:
                c.execute(q)
            c.execute("commit")
            c.close()
        
        with self.__get_pg_cursor() as c:
            for q in insert_queries:
                c.execute(q[0], (q[1],))
            c.execute("commit")
            c.close()
        # with Pool(processes=4) as pool:  
        #     pool.map(self.write_with_data, insert_queries)
        nudf_file = "/home/pyc/workspace/aqua_v2/aqua_extension/src/tool_sql/drop.sql"
        with open(nudf_file, 'a+') as file:
            file.write("\n".join(delete_quereis))
       
 
    def delete_batch_table(self, table_names):
        delete_query = 'DROP TABLE IF EXISTS "{}" CASCADE'                                                                                                                                                                                                                       
        table_names = [table_name.lower() for table_name in table_names]
        print(table_names)
        with self.__get_pg_cursor() as c:
            for table_name in table_names:
                delete_query = f'DROP TABLE IF EXISTS "{table_name}" CASCADE;'
                c.execute(delete_query)
            c.execute("commit") 

       
        
