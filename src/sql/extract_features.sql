-- Query de extração para treino (features + target).
-- Lida por AthenaClient.execute_query via src/config/feature.py (project.yml: data.query_file).
SELECT *
FROM todos_data_lake.table_name
