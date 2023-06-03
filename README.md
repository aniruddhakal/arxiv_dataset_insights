# arxiv_dataset_insights

### Description
TODO

### Assumed requirements
- Docker is installed, and appropriate permissions are granted for running this project.

### Steps to initialize dataset
1. Download and unzip dataset and place it in `dataset` directory
   - Dataset url
   ```bash
   https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
   ```
   - Place `arxiv-metadata-oai-snapshot.json` file in `dataset` directory
   
1. Initialize Mongodb collection, so that it's easy to access small portion of dataset as needed rather than loading everything in memory.
   - Run init_database service, wait for it to finish
   ```bash
    docker-compose up init_database
    ```
 
1. Run app

1. View jupyter notebooks
   - Launch jupyter notebooks instance
   ```bash
   docker-compose up -d notebooks
   ```
   
   - Access it from browser
   ```
   http://localhost:8888
   ```
   
### References
1. BERTopic - https://maartengr.github.io/BERTopic/