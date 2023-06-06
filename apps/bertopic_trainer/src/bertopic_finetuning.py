import yaml
from cuml import UMAP
from cuml.cluster.hdbscan import HDBSCAN
from optuna.trial import Trial
from util import *
from uuid import uuid4
import optuna
import re
import logging
from pathlib import Path
import argparse
import numpy as np
from typing import List
import torch
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from logging import Logger, StreamHandler
from cuml import HDBSCAN

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from hyperparameters import BertopicHyperparameters


class BERTopicTrainer:
    def __init__(self, config: dict, logger: Logger, ):
        self.config = config
        self.study_name = config['study_name']

        self.dataset_path = Path(self.config['dataset_path'])
        self.models_path = Path(self.config['models_path'])
        self.cache_dir = self.dataset_path / "cache_dir"
        self.logger = logger

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.dataset_index = config.get("dataset_index", 1)
        self.batch_size = config.get("batch_size", 384)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.datasets = self.load_datasets(dataset_index=self.dataset_index)
        self.dataset_keys = list(self.datasets.keys())

        self.best_model = None
        self.best_score = 1_000_000
        self.best_validation_score = 1_000_000

        self.hyperparameters = BertopicHyperparameters()

        # TODO should go away once optuna is integrated
        self.scores = {
            'coherence': [],
            'diversity': [],
        }

    def extract_or_load_embeddings(self, split_name: str):
        embeddings = self.load_embeddings(split_name)
        sentences = self.datasets[split_name]['abstract'][:]

        generating_fresh = False

        if not (embeddings is not None and embeddings.shape[0] == len(sentences)):
            generating_fresh = True

            sentence_transformer = SentenceTransformer(model_name_or_path=self.hyperparameters.model_name,
                                                       device=self.device)

            embeddings = sentence_transformer.encode(sentences=sentences,
                                                     batch_size=self.batch_size,
                                                     device=self.device,
                                                     convert_to_numpy=True,
                                                     show_progress_bar=True)

        if generating_fresh:
            self.save_embeddings(split_name, embeddings=embeddings)

        return embeddings

    def set_hyperparameters(self, trial: Trial):
        self.hyperparameters = BertopicHyperparameters(config=self.config, trial=trial, logger=logger)

    def save_embeddings(self, split_name: str, embeddings: np.ndarray):
        embeddings_filename = self.get_embeddings_filename(split_name)
        np.save(embeddings_filename, embeddings)

    def load_embeddings(self, split_name: str):
        embeddings_filename = self.get_embeddings_filename(split_name)
        embeddings = None

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

    def load_datasets(self, dataset_index: int, load_train: bool = True, load_validation: bool = True,
                      load_test: bool = True):
        datasets = {}

        if load_train:
            train_dataset = \
                load_dataset('parquet', data_files=[str(self.dataset_path / f"train_df_dataset_{dataset_index}.pq")],
                             cache_dir=self.cache_dir)['train']
            datasets['train'] = train_dataset

        if load_validation:
            validation_dataset = \
                load_dataset('parquet',
                             data_files=[str(self.dataset_path / f"validation_df_dataset_{dataset_index}.pq")],
                             cache_dir=self.cache_dir)['train']
            datasets['validation'] = validation_dataset

        if load_test:
            test_dataset = \
                load_dataset('parquet', data_files=[str(self.dataset_path / f"test_df_dataset_{dataset_index}.pq")],
                             cache_dir=self.cache_dir)['train']
            datasets['test'] = test_dataset

        return datasets

    def init_model(self):
        hp = self.hyperparameters
        count_vectorizer = CountVectorizer(
            max_features=hp.max_features, max_df=hp.max_df, min_df=hp.min_df,
            ngram_range=hp.n_gram_range, lowercase=hp.lowercase,
            stop_words=list(hp.stop_words)
            if not isinstance(hp.stop_words, list)
            else hp.stop_words,
        )

        umap_model = UMAP(
            n_neighbors=hp.n_neighbors, n_components=hp.n_components, metric=hp.umap_metric, n_epochs=hp.n_epochs, learning_rate=hp.learning_rate, min_dist=hp.min_dist,
            random_state=hp.random_state
        )

        hdbscan_model = HDBSCAN(min_cluster_size=hp.min_cluster_size,
                                cluster_selection_epsilon=hp.cluster_selection_epsilon, metric=hp.hdbscan_metric,
                                cluster_selection_method=hp.cluster_selection_method)

        # Create BERTopic with current number of categories
        model = BERTopic(
            nr_topics=hp.nr_topics,
            n_gram_range=hp.n_gram_range,
            vectorizer_model=count_vectorizer,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model
        )

        return model

    def _get_finetuning_study_name(self):
        study_name = self.models_path / self.study_name / f"finetuning_study.sql"
        study_name.parent.mkdir(parents=True, exist_ok=True)
        return str(study_name)

    def objective(self, trial: Trial):
        self.set_hyperparameters(trial=trial)
        train_score, validation_score = self.finetune()
        trial.report(validation_score, step=1)

        return validation_score

    def test(self):
        # test
        sentences_test = self.datasets['test']['abstract'][:]
        test_embeddings = self.extract_or_load_embeddings(split_name='test')
        topics, _ = self.best_model.transform(documents=sentences_test, embeddings=test_embeddings)
        coherence_score_test = self.calculate_coherence_score(model=self.best_model, sentences=sentences_test)
        self.logger.info(f'Test Score: {coherence_score_test}')

    def finetune(self):
        hp = self.hyperparameters

        train_embeddings = self.extract_or_load_embeddings(split_name='train')
        validation_embeddings = self.extract_or_load_embeddings(split_name='validation')

        # fetch texts
        sentences_train = self.datasets['train']['abstract'][:]
        sentences_validation = self.datasets['validation']['abstract'][:]

        self.logger.debug(f"Processing num_categories = {hp.nr_topics}")
        try:
            model = self.init_model()

            # fit to model
            topics, _ = model.fit_transform(documents=sentences_train, embeddings=train_embeddings)
            coherence_score_train = self.calculate_coherence_score(model=model, sentences=sentences_train)

            # validate
            topics, _ = model.transform(documents=sentences_validation, embeddings=validation_embeddings)
            coherence_score_validation = self.calculate_coherence_score(model=model, sentences=sentences_validation)

            self.append_score(metric_name='coherence', score=coherence_score_train)
            self.append_score(metric_name='coherence', score=coherence_score_validation)

            self.replace_best_model(model, train_score=coherence_score_train,
                                    validation_score=coherence_score_validation)

            return coherence_score_train, coherence_score_validation

        except Exception as e:
            raise Exception(f"Trial for num_categories {hp.nr_topics} failed with errror - {e}")

    def replace_best_model(self, model, train_score, validation_score):
        if train_score < self.best_score and validation_score < self.best_validation_score:
            self.best_model = model
            self.best_score = train_score
            self.best_validation_score = validation_score

    def save_model(self, model: BERTopic):
        study_name = self._get_finetuning_study_name()
        model_name = f"{str(Path(study_name).parent)}/model.bin"
        self.logger.debug(f"Saved model at {model_name}")
        model.save(model_name)
        self.logger.info(f"Saved model at {model_name}")

    def main(self):
        study_name = self._get_finetuning_study_name()
        optuna_study = optuna.create_study(direction='minimize', study_name=study_name, load_if_exists=True)
        optuna_study.optimize(self.objective, n_trials=self.config.get('n_trials', 5))

        self.logger.info(f"Best Trial - {optuna_study.best_trial}")
        self.logger.info(f"Best Value - {optuna_study.best_value}")
        self.logger.info(f"Best Params - {optuna_study.best_params}")

        if self.best_model is not None:
            self.logger.info(f"Saving best model")
            self.save_model(model=self.best_model)
        else:
            self.logger.warning(f"No best model was set, nothing to save")

    def calculate_coherence_score(self, model, sentences: List[str]):
        # --------------------------------
        # for calculating coherence score
        cleaned_docs = model._preprocess_text(sentences)
        analyzer = model.vectorizer_model.build_analyzer()
        tokens = [analyzer(doc) for doc in cleaned_docs]

        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]

        topics = model.get_topics()
        topics.pop(-1, None)

        topic_words = [
            [
                words for words, _ in model.get_topic(topic)
            ]
            for topic in range(len(set(topics)) - 1)
        ]
        # --------------------------------

        coherence_model = CoherenceModel(topics=topic_words,
                                         texts=tokens,
                                         corpus=corpus,
                                         dictionary=dictionary,
                                         coherence='c_v')

        coherence_score = coherence_model.get_coherence()

        return coherence_score

    def append_score(self, metric_name: str, score):
        # TODO change to log scores to optuna study
        scores_list = self.scores.get(metric_name)

        if scores_list is None:
            raise Exception("Invalid scoring metric")

        scores_list.append(score)
        self.scores[metric_name] = scores_list


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(args.config_file)

    logfile_name = get_logfile_name(config=config)

    logger = initialize_logger(logfile_name=logfile_name, log_level=config['logging_level'])
    trainer = BERTopicTrainer(config=config, logger=logger)
    trainer.finetune()
    trainer.test()
