# arxiv_dataset_insights

## The work is still under progress


## Documentation Navigation
- `ReadMe.md` (you are here)
  - Covers details on Installing requirements, and initializing dataset.
  - Then redirects you to `ResearchSteps.md` file.
- `ResearchSteps.md`
  - Covers all steps from literature review, data preprocessing, EDA and insights.
  - Then redirects you to `HyperparameterTuning.md` file.
- `HyperparameterTuning.md`
  - Covers Hyperparameters Tuning in details, covering several studies and including decisions to continue with certain hyperparameters,
  for both Topic Model and Classifier.
  - Then redirects you back to this page to view Evaluations details, where later sections continue with inference app, conclusions, and more.


## All Steps
- [[TODO] Description](#description)
- [Assumed Requirements](#assumed-requirements)
- [Initialize Dataset](#steps-to-initialize-dataset)
- [Research Steps](#research-steps)
- [Running Classifier Evaluations](#running-classifier-evaluations)
  - [Evaluator YAML Configuration](#evaluator-yaml-configuration)
  - [Key YAML Elements Explained](#key-yaml-elements-explained)
- [[TODO] Inference App / Endpoint](#inference-app--endpoint)
- [[TODO] Conclusions](#conclusions)
- [[TODO] Data Visualizations](#data-visualizations)
- [Issues Faced](#issues-faced)


### Description
TODO

### Assumed Requirements
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

### Research Steps
Research steps are comprehensively covered in file [ResearchSteps.md](ResearchSteps.md)
Don't worry, you'll be redirected to section after this one.


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


## Inference App / Endpoint
- ##### TODO


## Conclusions
### TODO
- Conclude findings when choosing models
- Conclusions on model choice
- Conclusions on BERTopic technique, and choosing best number of topics
- Conclusions on reproducibility of BERTopic, esp. on GPU
- Conclusions on dim reduction, and clustering algorithm's maturity on ability to work in batches on GPU's and not occupying the whole memory.
- Conclusions on Classifier and results, and need for better metrics
#### TODO choose best number of topics. 


## Data Visualizations
### TODO


## Issues faced

- Installation related issues
- Issue with mounting GPU to docker container **[unresolved]**
- Issues with extracting Embeddings for Clustering, the model, size, and time limit considerations, computations limits
  on local machine
- Issues while installing cuML library
- Issues faced while trying evaluation metrics from OCTIS
- GPU memory leaks and inability to proceed with fine-tuning of larger datasets I selected - datasets 1, 2, 3, 4
    - Although I did manage to train 1 model on dataset 1, for which, the training data is approx 40% of the original
      dataset.
        - That too was achieved by only reducing precision of embeddings to float16.
        - Clearing GPU context memory after intermediate steps of dimensionality reduction, and clustering, but not as
          much for bigger datasets.
- Difficulty in reproducing BERTopic results, at least in my case. This is potentially due to using GPU based
  computations across 80% processing.