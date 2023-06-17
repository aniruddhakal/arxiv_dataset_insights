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
- [Running Classifier Evaluations](#running-classifier-evaluations)
  - [Evaluator YAML Configuration](#evaluator-yaml-configuration)
  - [Key YAML Elements Explained](#key-yaml-elements-explained)


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

### Running Classifier Evaluations
- Classifier Evaluator program is available at location - `<project_root>/apps/classifier/src/classifier_evaluator.py`
- Evaluator uses `yaml` config file

- Here's how you run the example evaluations
```bash
# example for dataset4
# assuming you are at the project root which is inside <arxiv_dataset_insights> directory
cd apps/classifier/src/
python classifier_evaluator.py ../config/evaluator_dataset5.yml
```

- Some evaluator results can be seen in these log files:
  - Evaluations for dataset 4 - `./apps/classifier/src/classifier_evaluator_dataset4.txt-1762023-195325.txt`
  - Evaluations for dataset 5 - `./apps/classifier/src/classifier_evaluator_dataset5.txt-1762023-195732.txt`

#### Evaluator YAML Configuration
```yaml
logging_dir: "../logs"
logfile_name: "classifier_evaluator_dataset5.txt"
logging_level: "INFO"

dataset_path: "../../../dataset"
models_path: "../../../models"

# This indicates the model chosen for evaluations is the best
# model saved during specified Optuna hyperparameter tuning study.
optuna_study_name: "classifier_dataset4_study1"

dataset_index: 5

# model settings
label_transformer: "multilabel_binarizer.pkl"
transformer_model_name: 'sentence-transformers/distilroberta-base-paraphrase-v1'
num_classes: 176
input_size: 768

batch_size: 1024
```
#### Key YAML Elements Explained
- `optuna_study_name`: This indicates the model chosen for evaluations is the best
model saved during specified Optuna hyperparameter tuning study.
- `dataset_index` - Checkout section [Dataset Creation](#dataset-creation) on how datasets are created.
- `label_transformer` - name of label_transformer that is present with the same name under `models_path` directory.
This model is of type `sklearn.preprocessing.MultiLabelBinarizer` and supports 176 unique classes extracted from `categories` from the original arxiv dataset.