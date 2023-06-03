import json
from pymongo import MongoClient
from logging import Logger, StreamHandler

logger = Logger(__name__)
logger.addHandler(StreamHandler())

db_connection_string = "mongodb://localhost:21000"
client = MongoClient(db_connection_string)
db = client['arxiv-db']
collection = db['arxiv-dataset-collection']

file_path = '/dataset/arxiv-metadata-oai-snapshot.json'

print_every = 100_000

try:
    with open(file_path, 'r') as file:
        n_written = 0
        counter = 0

        for line in file:
            json_line = json.loads(line)
            collection.insert_one(document=json_line)
            n_written += 1

            if counter == print_every:
                counter = 0
                logger.info(f"Processed {n_written} records")
except Exception as e:
    logger.error(f"Exception occurred while reading file from {file_path}, Exception - {e}")
    raise e

logger.info(
    f"Successfully written all records to MongoDB collection,"
    f"use this connection string to access records: {db_connection_string}")

client.close()
