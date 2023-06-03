# arxiv_dataset_insights

## The work is still under progress


### All Steps
- [[In Progress] Research Steps](ResearchSteps.md)
- [Initialize Dataset (MongoDB)](#steps-to-initialize-dataset)
- [EDA and Creating Train, Validation, Test splits](#eda-and-creating-train-validation-test-splits)
- ###### [Currently under progress > Hyperparameter Tuning](#hyperparameter-tuning)
- [TODO Viewing Finetuning Results with Optuna Dashboard](#viewing-finetuning-results-with-optuna-dashboard)
- [TODO Inference App / Endpoint](#inference-app--endpoint)
- [TODO Model Choice Decisions](#model-choice-decisions)
- [TODO Comprehensive Documentation]()


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

### EDA and Creating Train, Validation, Test splits
1. Launch jupyter notebooks at project root directory
   ```bash
   cd <project_root>
   jupter notebook
   ```
1. Run `./apps/notebooks/src/EDA.ipynb`
   2. Along with some EDA, this notebook also generates Train, Validation, and Test dataset splits. 
   3. ##### TODO later to be converted to script

### Hyperparameter Tuning
  - Finetuning can be run as follows
  ```bash
   cd <project_root>/apps/bertopic_trainer
   python bertopic_finetuning.py config/bertopic_finetuning_config_dataset_1.yml 
   ```

### Viewing Finetuning Results with Optuna Dashboard
- ##### TODO

### Inference App / Endpoint
- ##### TODO

### Model Choice Decisions 
- ##### TODO


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