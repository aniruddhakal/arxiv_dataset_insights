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
- All the requirements to run the corresponding sub-modules are provided within requirements.txt file at the root of that module.
  - Hence, consider creating python virtual environment and install all the requirements from requirements.txt file.
- **[skip this step]** Docker is installed, and appropriate permissions are granted for running this project.
  - Plan to use docker for everything was dropped because I couldn't fully figure out how to access GPU from docker instance.
  - Using things through docker containers, would work, but not using GPU would slow things down significantly.

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
   
    - Unfortunately, I haven't fully figured out how to attach existing GPU to any docker instance, so prefer running jupyter notebooks rather from local machine directly.
   
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
  - Evaluations for dataset 4 - `./apps/classifier/src/classifier_evaluator_dataset4-1762023-195325.txt`
  - Evaluations for dataset 5 - `./apps/classifier/src/classifier_evaluator_dataset5-1762023-195732.txt`

#### Evaluator YAML Configuration
```yaml
logging_dir: "../logs"
logfile_name: "classifier_evaluator_dataset5"
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

#### Classifier Evaluation Results

##### Dataset 4
|       Threshold       | Top-1-Overlap Accuracy |
|:---------------------:|:----------------------:|
|   <th> Train </th>    |                        |
|         0.05          |         0.9570         |
|          0.1          |         0.9239         |
|          0.2          |         0.8627         |
|          0.3          |         0.8017         |
| <th> Validation </th> |                        |
|         0.05          |         0.9434         |
|          0.1          |         0.9044         |
|          0.2          |         0.8369         |
|          0.3          |         0.7737         |
|  <th> Test </th>      |                        |
|         0.05          |         0.9432         |
|          0.1          |         0.9044         |
|          0.2          |         0.8373         |
|          0.3          |         0.7738         |


##### Dataset 5
|       Threshold       | Top-1-Overlap Accuracy |
|:---------------------:|:----------------------:|
|   <th> Train </th>    |                        |
|         0.05          |         0.9568         |
|          0.1          |         0.9236         |
|          0.2          |         0.8632         |
|          0.3          |         0.8016         |
| <th> Validation </th> |                        |
|         0.05          |         0.9533         |
|          0.1          |         0.9185         |
|          0.2          |         0.8532         |
|          0.3          |         0.7897         |
|  <th> Test </th>      |                        |
|         0.05          |         0.9527         |
|          0.1          |         0.9178         |
|          0.2          |         0.8531         |
|          0.3          |         0.7919         |
