import yaml
from util import *
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

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from hyperparameters import BertopicHyperparameters


class BERTopicTrainer:
    def __init__(self, config: dict, logger: Logger, ):
        self.config = config
        self.hyperparameters = BertopicHyperparameters(config=config, logger=logger)

        self.dataset_path = Path(self.config['dataset_path'])
        self.cache_dir = self.dataset_path / "cache_dir"
        self.logger = logger

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_index = config.get("dataset_index", 1)
        self.model_name = config.get("model_name", "distilbert-base-nli-mean-tokens")
        self.batch_size = config.get("batch_size", 384)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.datasets = self.load_datasets(dataset_index=self.dataset_index)
        self.dataset_keys = list(self.datasets.keys())

        self.best_model = None
        self.best_score = 1_000_000
        self.best_validation_score = 1_000_000

        # TODO test following
        #   - distilbert-base-nli-mean-tokens
        #   - 'sentence-transformers/all-mpnet-base-v2'

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

            sentence_transformer = SentenceTransformer(model_name_or_path=self.model_name, device=self.device)

            embeddings = sentence_transformer.encode(sentences=sentences,
                                                     batch_size=self.batch_size,
                                                     device=self.device,
                                                     convert_to_numpy=True,
                                                     show_progress_bar=True)

        if generating_fresh:
            self.save_embeddings(split_name, embeddings=embeddings)

        return embeddings

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
            f"split-{split_name}_dataset-{self.dataset_index}_model-{self.model_name}_embeddings.npy"
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

    def init_model(self, num_categories: int):
        hp = self.hyperparameters
        # TODO call suggest_hyperparameters() method, pass optuna trial as input
        count_vectorizer = CountVectorizer(
            max_features=hp.max_features, max_df=hp.max_df, min_df=hp.min_df,
            ngram_range=hp.n_gram_range, lowercase=hp.lowercase,
            stop_words=list(hp.stop_words)
            if not isinstance(hp.stop_words, list)
            else hp.stop_words
        )

        # Create BERTopic with current number of categories
        model = BERTopic(
            nr_topics=num_categories,
            vectorizer_model=count_vectorizer,
            n_gram_range=hp.n_gram_range
        )

        return model

    def train(self):
        # TODO implement
        raise NotImplementedError()

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

        # TODO init and start optuna study
        for num_categories in range(hp.min_categories, hp.max_categories + 1):
            # TODO log optuna study trial start
            self.logger.debug(f"Processing num_categories = {num_categories}")
            try:
                model = self.init_model(num_categories)

                # fit to model
                topics, _ = model.fit_transform(documents=sentences_train, embeddings=train_embeddings)
                coherence_score_train = self.calculate_coherence_score(model=model, sentences=sentences_train)

                # validate
                topics, _ = model.transform(documents=sentences_validation, embeddings=validation_embeddings)
                coherence_score_validation = self.calculate_coherence_score(model=model, sentences=sentences_validation)

                # TODO log train and validation scores to optuna study
                #   figure out appropriate ways to log train, test, and validation scores to optuna
                self.append_score(metric_name='coherence', score=coherence_score_train)
                self.append_score(metric_name='coherence', score=coherence_score_validation)

                self.replace_best_model(model, train_score=coherence_score_train,
                                        validation_score=coherence_score_validation)

                # TODO optuna trial finish update
            except Exception as e:
                # TODO optuna trial fail update
                raise Exception(f"Trial for num_categories {num_categories} failed with errror - {e}")

    def replace_best_model(self, model, train_score, validation_score):
        if train_score < self.best_score and validation_score < self.best_validation_score:
            self.best_model = model
            self.best_score = train_score
            self.best_validation_score = validation_score

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
