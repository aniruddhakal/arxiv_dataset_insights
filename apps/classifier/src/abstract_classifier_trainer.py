import gc
import pickle
from tqdm import tqdm
import re
from logging import Logger
from typing import List, Union, Dict
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import optuna
import torch
from datasets import load_dataset
from gensim.corpora import Dictionary
from optuna.trial import Trial
from sentence_transformers import SentenceTransformer

from hyperparameters import ClassifierHyperparameters
from arxiv_dataset import ArxivDataset
from abstract_classification_models import ArxivAbstractClassifier
from util import *


class AbstractClassificationTrainer:
    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.study_name = config['study_name']
        self.study_storage_name = config['study_storage_filename']

        self.dataset_path = Path(self.config['dataset_path'])
        self.models_path = Path(self.config['models_path'])
        self.cache_dir = self.dataset_path / "cache_dir"
        self.logger = logger

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.dataset_index = config.get("dataset_index", 1)
        self.batch_size = config.get("batch_size", 384)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.datasets = {}
        self.dataset_keys = list(self.datasets.keys())
        self.embeddings = {}

        self.best_score = 9999
        self.best_validation_score = 9999

        self.hyperparameters = None

        label_transformer_name = self.config['label_transformer']
        self.multilabel_binarizer = None
        with open(self.models_path / label_transformer_name, 'rb') as f:
            self.multilabel_binarizer = pickle.load(f)
        if not self.multilabel_binarizer:
            raise Exception(f"Label Transformer found to be None")
        self.logger.debug(f"Loaded Label Transformer with {len(self.multilabel_binarizer.classes_)} classes")

    def extract_or_load_embeddings(self, split_name: str, cache=True):
        # TODO make caching explicitly configurable through config file
        self.logger.debug(f"Loading embeddings for split {split_name}")
        embeddings = self.load_embeddings(split_name)
        sentences = self.datasets[split_name]['abstract'][:]

        if not (embeddings is not None and embeddings.shape[0] == len(sentences)):
            self.logger.debug(f"Generating fresh embeddings for {len(sentences)} sentences")

            sentence_transformer = SentenceTransformer(model_name_or_path=self.hyperparameters.model_name,
                                                       device=self.device)

            embeddings = sentence_transformer.encode(sentences=sentences,
                                                     batch_size=self.batch_size,
                                                     device=self.device,
                                                     convert_to_numpy=True,
                                                     show_progress_bar=True)

            self.save_embeddings(split_name, embeddings=embeddings)

            if cache:
                self.embeddings[split_name] = embeddings

        return embeddings

    def set_hyperparameters(self, trial: Trial):
        self.hyperparameters = ClassifierHyperparameters(config=self.config, logger=self.logger, trial=trial)

    def save_embeddings(self, split_name: str, embeddings: np.ndarray):
        embeddings_filename = self.get_embeddings_filename(split_name)
        np.save(embeddings_filename, embeddings)
        self.logger.debug(f"Saved embeddings of shape {embeddings.shape} in file {embeddings_filename}")

    def load_embeddings(self, split_name: str):
        embeddings = self.embeddings.get(split_name)

        if embeddings is None:
            embeddings_filename = self.get_embeddings_filename(split_name)

            if Path(embeddings_filename).exists():
                try:
                    embeddings = np.load(embeddings_filename)
                except FileNotFoundError as e:
                    self.logger.error(f"Expected file named {embeddings_filename} was not found")

        return embeddings

    def get_embeddings_filename(self, split_name):
        return str(
            self.dataset_path /
            f"split-{split_name}_dataset-{self.dataset_index}_model-{self.hyperparameters.model_normalized_name}_embeddings.npy"
        )

    def load_dataset(self, split: str, lemmatized: bool = False, cache=True, dataset_index: int = None):
        if dataset_index is None:
            dataset_index = self.dataset_index
        prefix = ""

        if lemmatized:
            prefix = "lemmatized_"

        cache_name = split
        if lemmatized:
            cache_name = f"{prefix}{cache_name}"

        dataset = self.datasets.get(cache_name)
        if dataset is None:
            dataset = \
                load_dataset('parquet',
                             data_files=[str(self.dataset_path / f"{prefix}{split}_df_dataset_{dataset_index}.pq")],
                             cache_dir=self.cache_dir)['train']

            if cache:
                self.datasets[cache_name] = dataset

        return dataset

    def load_datasets(self, dataset_index: int, load_train: bool = True, load_validation: bool = True,
                      load_test: bool = True):

        if load_train:
            self.load_dataset(dataset_index=dataset_index, split='train', lemmatized=False)

        if load_validation:
            self.load_dataset(dataset_index=dataset_index, split='validation', lemmatized=False)

        if load_test:
            self.load_dataset(dataset_index=dataset_index, split='test', lemmatized=False)

    def init_model(self):
        self.logger.debug(f"Initializing model")
        hp = self.hyperparameters

        # default model - 'dense'
        model = ArxivAbstractClassifier(input_size=hp.input_size, num_classes=hp.num_classes)

        # TODO add more models here
        if hp.classifier_model_name == 'something_else':
            model = ArxivAbstractClassifier(input_size=hp.input_size, num_classes=hp.num_classes)

        self.logger.debug(f"Model initialized")
        return model

    def _get_finetuning_study_details(self):
        postfix = '.sql'

        if self.study_storage_name.endswith('.sql'):
            postfix = ''

        study_storage = self.models_path / "finetuning_studies" / f"{self.study_storage_name}{postfix}"
        study_storage.parent.mkdir(parents=True, exist_ok=True)

        return {
            "storage": study_storage,
            "study_name": self.study_name
        }

    def objective(self, trial: Trial):
        try:
            self.set_hyperparameters(trial=trial)

            train_embeddings = self.extract_or_load_embeddings(split_name='train')
            validation_embeddings = self.extract_or_load_embeddings(split_name='validation')

            train_dataset = self.load_dataset(split='train')
            validation_dataset = self.load_dataset(split='validation')

            train_y = self.transform_labels(train_dataset['categories_list'])
            validation_y = self.transform_labels(validation_dataset['categories_list'])

            train_score, validation_score, all_train_preds, all_train_y, all_val_preds, all_val_y = self.finetune(
                trial=trial,
                train_embeddings=train_embeddings,
                validation_embeddings=validation_embeddings,
                train_y=train_y,
                validation_y=validation_y)

            # train_scores = self.run_evaluations(split="train", labels=all_train_y, predictions=all_train_preds)
            # validation_scores = self.run_evaluations(split='validation', labels=all_val_y, predictions=all_val_preds)

            # self.report_scores_to_optuna_trial(split='train', scores=train_scores, trial=trial)
            # self.report_scores_to_optuna_trial(split='validation', scores=validation_scores, trial=trial)
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {e}")
            trial.set_user_attr(key="failure_reason", value=str(e))
            raise optuna.exceptions.TrialPruned() from e
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return validation_score

    def validation_loop(self, model: nn.Module, validation_dataloader: DataLoader):
        model.eval()
        validation_loss = 0.0

        predictions = []
        val_loop_y = []

        criterion = nn.BCELoss()
        criterion = self.move_to_device(criterion)

        for inputs, labels in tqdm(validation_dataloader):
            val_loop_y.extend(labels.tolist())
            with torch.no_grad():
                inputs = self.move_to_device(inputs)
                labels = self.move_to_device(labels)

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)
                # predictions.extend(outputs.tolist())

                validation_loss += loss.item()

        validation_loss = validation_loss / len(validation_dataloader)
        return validation_loss, predictions, val_loop_y

    def move_to_device(self, x):
        x = x.to(self.device)
        return x

    def training_loop(self, model: nn.Module, train_dataloader: DataLoader, criterion: nn.modules.loss._Loss,
                      optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler):
        model.train()
        running_loss = 0.0

        predictions = []
        train_loop_y = []

        for inputs, labels in tqdm(train_dataloader):
            # train_loop_y.extend(labels.tolist())
            # Zero the gradients
            optimizer.zero_grad()

            inputs = self.move_to_device(inputs)
            labels = self.move_to_device(labels)

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            # predictions.extend(outputs.tolist())

            # Backward pass
            loss.backward()

            # Update the model's parameters
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        scheduler.step()

        # Compute the average loss for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        return epoch_loss, optimizer, scheduler, predictions, train_loop_y

    def transform_labels(self, labels: List[List[str]]):
        y = self.multilabel_binarizer.transform(labels)
        return y

    def delete_datasets(self, splits: List[str] = None, all: bool = False):
        if all:
            splits = list(self.datasets.keys())

        if not splits:
            self.logger.info(f"No databases to delete")

        self.logger.debug(f"Cleaning up space used by dataset splits - {splits}")
        for key in splits:
            del self.datasets[key]

        gc.collect()

    def finetune(self, trial: Trial, train_embeddings: np.array, validation_embeddings: np.array, train_y: np.array,
                 validation_y: np.array):
        hp = self.hyperparameters
        self.logger.info(f"Using model {hp.model_name}")

        self.delete_datasets(all=True)

        # Create a custom dataset instance
        train_dataset = ArxivDataset(train_embeddings, train_y.astype(np.float32))
        validation_dataset = ArxivDataset(validation_embeddings, validation_y.astype(np.float32))

        train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2)
        validation_dataloader = DataLoader(validation_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=2)

        model = self.init_model()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp.scheduler_step_size, gamma=hp.scheduler_gamma)

        model = self.move_to_device(model)
        criterion = self.move_to_device(criterion)

        all_training_predictions = []
        all_train_y = []
        all_validation_predictions = []
        all_val_y = []

        try:
            training_loss = 999
            validation_loss = 999

            for epoch in tqdm(range(1, hp.n_epochs + 1)):
                self.logger.info(f"Epoch {epoch}")
                training_loss, optimizer, scheduler, training_predictions, train_loop_y = self.training_loop(model=model,
                                                                                               train_dataloader=train_dataloader,
                                                                                               criterion=criterion,
                                                                                               optimizer=optimizer,
                                                                                               scheduler=scheduler)
                # all_training_predictions.extend(training_predictions)
                # all_train_y.extend(train_loop_y)

                validation_loss, validation_predictions, val_loop_y = self.validation_loop(model=model,
                                                                               validation_dataloader=validation_dataloader)
                trial.report(value=validation_loss, step=epoch)
                # all_validation_predictions.extend(validation_predictions)
                # all_val_y.extend(val_loop_y)

                self.logger.info(f"Training Loss: {training_loss}, Validation Loss: {validation_loss}")

                # evaluations
                # train_scores = self.run_evaluations(split='train', labels=train_y, predictions=training_predictions)
                # validation_scores = self.run_evaluations(split='validation', labels=validation_y,
                #                                          predictions=validation_predictions)
                #
                # self.report_scores_to_optuna_trial(split=f'epoch_{epoch}_train', scores=train_scores, trial=trial)
                # self.report_scores_to_optuna_trial(split=f'epoch_{epoch}_validation', scores=validation_scores,
                #                                    trial=trial)

                self.replace_best_model(model=model, train_score=training_loss, validation_score=validation_loss)

            self.logger.info(f"Finished Training")

            return training_loss, validation_loss, all_training_predictions, all_train_y, all_validation_predictions, all_val_y
        except Exception as e:
            raise Exception(f"Trial failed with error - {e}")

    def report_scores_to_optuna_trial(self, split: str, scores: Dict, trial: Trial):
        for key in scores.keys():
            # reported as train_accuracy, given 'train' is the split name
            trial.set_user_attr(f"{split}_{key}", scores.get(key))

    def run_evaluations(self, split: str, labels: Union[List, np.array], predictions: Union[List, np.array]):
        from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, classification_report

        if isinstance(labels, List):
            labels = np.array(labels)

        if isinstance(predictions, List):
            predictions = np.array(predictions)

        self.logger.info(f"{split} evaluations")

        # accuracy
        def macro_average_accuracy(y_true, y_pred):
            num_labels = y_true.shape[1]  # Number of labels
            accuracies = []

            for label in range(num_labels):
                correct = (y_true[:, label] == y_pred[:, label]).sum()
                total = y_true.shape[0]
                accuracy = correct / total
                accuracies.append(accuracy)

            macro_avg_accuracy = sum(accuracies) / num_labels
            return macro_avg_accuracy

        accuracy = macro_average_accuracy(y_true=labels, y_pred=predictions)
        self.logger.info(f"Accuracy: {accuracy}")

        # hamming loss
        hamming = hamming_loss(y_true=labels, y_pred=predictions)
        self.logger.info(f"Hamming Loss: {hamming}")

        # precision
        precision = precision_score(y_true=labels, y_pred=predictions, average='macro')
        self.logger.info(f"Precision Score: {precision}")

        # recall
        recall = recall_score(y_true=labels, y_pred=predictions, average='macro')
        self.logger.info(f"Recall Score: {recall}")

        # f1_macro
        f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
        self.logger.info(f"F1-macro: {f1_macro}")

        # classification report
        logger.info(f"Classification Report: {classification_report(y_true=labels, y_pred=predictions)}")

        return {
            "accuracy": accuracy,
            "hamming": hamming,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
        }

    def test(self):
        hp = self.hyperparameters
        self.logger.info(f"Testing model")

        test_embeddings = self.extract_or_load_embeddings(split_name='test')

        test_dataset = self.load_dataset(split='test')
        test_y = self.transform_labels(test_dataset['categories_list'])
        self.delete_datasets(all=True)

        # Create a custom dataset instance
        test_dataset = ArxivDataset(test_embeddings, test_y.astype(np.float32))
        test_dataloader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2)

        model = self.load_best_model()
        model = self.move_to_device(model)

        try:
            test_loss, test_predictions, test_loop_y = self.validation_loop(model=model, validation_dataloader=test_dataloader)
            self.logger.info(f"Test Loss: {test_loss}")

            # self.run_evaluations(split='test', labels=test_loop_y, predictions=test_predictions)
        except Exception as e:
            raise Exception(f"Error occurred while running test - {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def replace_best_model(self, model, train_score, validation_score):
        if train_score < self.best_score and validation_score < self.best_validation_score:
            self.logger.info(
                f"Replacing best model."
                f" Old validation score: {self.best_validation_score},"
                f" New validation score: {validation_score}")

            self.save_model(model=model)

            self.best_score = train_score
            self.best_validation_score = validation_score

    def save_model(self, model: nn.Module, final_model=False):
        study_details = self._get_finetuning_study_details()
        study_name = study_details.pop('study_name')
        optuna_storage = study_details.pop("storage")

        target_dir = self.models_path / study_name if final_model else optuna_storage.parent / study_name
        target_dir.mkdir(parents=True, exist_ok=True)

        model_name = f"{str(target_dir)}/best_model.bin"

        self.logger.debug(f"Saving model at {model_name}")
        torch.save(model.state_dict(), model_name)
        self.logger.debug(f"Saved model at {model_name}")

    def load_best_model(self):
        study_details = self._get_finetuning_study_details()
        study_name = study_details.pop('study_name')
        optuna_storage = study_details.pop('storage')

        target_dir = optuna_storage.parent / study_name
        target_dir.mkdir(parents=True, exist_ok=True)

        model_name = f"{str(target_dir)}/best_model.bin"
        model = ArxivAbstractClassifier(input_size=self.hyperparameters.input_size,
                                        num_classes=self.hyperparameters.num_classes)
        model.load_state_dict(torch.load(model_name))
        return model

    def main(self):
        # load key datasets
        self.load_datasets(dataset_index=self.dataset_index, load_test=False)

        study_details = self._get_finetuning_study_details()
        study_name = study_details.pop('study_name')
        storage = study_details.pop("storage")
        optuna_study = optuna.create_study(direction='minimize', storage=f"sqlite:///{str(storage)}",
                                           study_name=study_name, load_if_exists=True)
        optuna_study.optimize(self.objective, n_trials=self.config.get('n_trials', 5))

        self.logger.info(f"Best Trial - {optuna_study.best_trial}")
        self.logger.info(f"Best Value - {optuna_study.best_value}")
        self.logger.info(f"Best Params - {optuna_study.best_params}")

    def drop_sep_tokens(self, sentences: List[str]) -> List[str]:
        processed_sentences = []

        for sentence in sentences:
            sentence = re.sub(r"(\s)?SEP(\s)?", " ", sentence, flags=re.IGNORECASE)
            processed_sentences.append(sentence)

        return processed_sentences


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(args.config_file)

    logfile_name = get_logfile_name(config=config)

    logger = initialize_logger(logfile_name=logfile_name, log_level=config['logging_level'])
    trainer = AbstractClassificationTrainer(config=config, logger=logger)
    trainer.main()
    trainer.test()
