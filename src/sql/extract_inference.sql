-- Query de extração para inferência batch (apenas features, sem target).
-- Lida por AthenaClient.execute_query via src/config/predict.py (project.yml: data.inference_query_file).
SELECT *
FROM todos_data_lake.table_name
