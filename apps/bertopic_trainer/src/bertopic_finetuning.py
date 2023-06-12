import gc
import re
from logging import Logger
from typing import List

import numpy as np
import optuna
import spacy
import torch
from custom_bertopic import CustomBERTopic
from bertopic import BERTopic
from cuml import HDBSCAN, UMAP
from datasets import load_dataset
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from optuna.trial import Trial
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from hyperparameters import BertopicHyperparameters
from util import *


class BERTopicTrainer:
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

        self.spacy_nlp = spacy.load("en_core_web_lg")

        self.best_score = -1
        self.best_validation_score = -1

        self.hyperparameters = None

        # TODO should go away once optuna is integrated
        self.scores = {
            'coherence': [],
            'diversity': [],
        }

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
        self.hyperparameters = BertopicHyperparameters(config=self.config, logger=self.logger, trial=trial)

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

        n_gram_range = (hp.n_gram_range_start, hp.n_gram_range_end)

        count_vectorizer = CountVectorizer(
            max_features=hp.max_features, max_df=hp.max_df, min_df=hp.min_df,
            ngram_range=n_gram_range, lowercase=hp.lowercase,
            stop_words=list(STOPWORDS),
        )

        umap_model = UMAP(
            n_neighbors=hp.n_neighbors, n_components=hp.n_components, metric=hp.umap_metric, n_epochs=hp.n_epochs,
            learning_rate=hp.learning_rate, min_dist=hp.min_dist,
            random_state=hp.random_state
        )

        hdbscan_model = HDBSCAN(min_cluster_size=hp.min_cluster_size,
                                cluster_selection_epsilon=hp.cluster_selection_epsilon, metric=hp.hdbscan_metric,
                                cluster_selection_method=hp.cluster_selection_method,
                                prediction_data=True)

        # Create BERTopic with current number of categories
        model = CustomBERTopic(
            logger=self.logger,
            nr_topics=hp.nr_topics,
            n_gram_range=n_gram_range,
            vectorizer_model=count_vectorizer,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            top_n_words=hp.top_n_words,
        )

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
            train_score, validation_score = self.finetune()
            trial.report(validation_score, step=1)
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {e}")
            trial.set_user_attr(key="failure_reason", value=str(e))
            raise optuna.exceptions.TrialPruned() from e

        return validation_score

    def test(self):
        # test
        test_dataset = self.load_dataset(split='test', lemmatized=False, cache=True)

        sentences_test = test_dataset['abstract'][:]
        self.logger.debug(f"Testing model performance on test dataset with {len(sentences_test)} samples")
        test_embeddings = self.extract_or_load_embeddings(split_name='test')

        self.logger.debug(f"Loading best model")
        model = self.load_best_model()
        self.logger.debug(f"Running inference on test dataset")
        topics, _ = model.transform(documents=sentences_test, embeddings=test_embeddings)
        del sentences_test
        del test_embeddings

        lemmatized_test_dataset = self.load_dataset(split='test', lemmatized=True, cache=True)
        lemmatized_test_sentences = lemmatized_test_dataset['abstract'][:]
        coherence_score_test = self.calculate_coherence_score(model=model,
                                                              sentences=lemmatized_test_sentences)
        del lemmatized_test_sentences
        self.logger.info(f'Test Score: {coherence_score_test}')

    def train_model(self, lemmatized_train_dataset, train_embeddings, lemmatized_validation_dataset,
                    validation_embeddings):
        model = self.init_model()

        # fit to model
        self.logger.debug(f"Starting training")
        # model.pa
        lemmatized_train_sentences = lemmatized_train_dataset['abstract'][:]
        topics, _ = model.fit_transform(documents=lemmatized_train_sentences,
                                        embeddings=train_embeddings.astype(np.float16))
        # model.partial_fit(documents=lemmatized_train_sentences, embeddings=train_embeddings)
        # model.get_topics()
        self.logger.debug(f"Finished training")

        coherence_score_train = self.calculate_coherence_score(model=model, sentences=lemmatized_train_sentences)
        del lemmatized_train_sentences

        # validate
        self.logger.debug(f"Running validations")
        lemmatized_validation_sentences = lemmatized_validation_dataset['abstract'][:]
        topics, _ = model.transform(documents=lemmatized_validation_sentences,
                                    embeddings=validation_embeddings.astype(np.float16))

        coherence_score_validation = self.calculate_coherence_score(model=model,
                                                                    sentences=lemmatized_validation_sentences)
        del lemmatized_validation_sentences

        self.append_score(metric_name='coherence', score=coherence_score_train)
        self.append_score(metric_name='coherence', score=coherence_score_validation)

        self.replace_best_model(model, train_score=coherence_score_train,
                                validation_score=coherence_score_validation)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return coherence_score_train, coherence_score_validation

    def finetune(self):
        hp = self.hyperparameters

        self.logger.info(f"Using model {hp.model_name}")
        train_embeddings = self.extract_or_load_embeddings(split_name='train')
        validation_embeddings = self.extract_or_load_embeddings(split_name='validation')

        # fetch texts
        lemmatized_train_dataset = self.load_dataset(split='train', lemmatized=True, cache=True)
        lemmatized_validation_dataset = self.load_dataset(split='validation', lemmatized=True, cache=True)

        self.logger.debug(f"Processing num_categories = {hp.nr_topics}")
        try:
            self.train_model(lemmatized_train_dataset=lemmatized_train_dataset, train_embeddings=train_embeddings,
                             lemmatized_validation_dataset=lemmatized_validation_dataset,
                             validation_embeddings=validation_embeddings)
        except Exception as e:
            raise Exception(f"Trial for num_categories {hp.nr_topics} failed with errror - {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def replace_best_model(self, model, train_score, validation_score):
        if train_score > self.best_score and validation_score > self.best_validation_score:
            self.logger.info(
                f"Replacing best model."
                f" Old validation score: {self.best_validation_score},"
                f" New validation score: {validation_score}")

            self.save_model(model=model)

            self.best_score = train_score
            self.best_validation_score = validation_score

    def save_model(self, model: BERTopic, final_model=False):
        # TODO also save count vectorizer,  umap, and hdbscan models (if necessary)
        study_details = self._get_finetuning_study_details()
        study_name = study_details.pop('study_name')
        optuna_storage = study_details.pop("storage")

        target_dir = self.models_path / study_name if final_model else optuna_storage.parent / study_name
        target_dir.mkdir(parents=True, exist_ok=True)

        model_name = f"{str(target_dir)}/best_model.bin"

        self.logger.debug(f"Saving model at {model_name}")
        model.save(model_name)
        self.logger.debug(f"Saved model at {model_name}")

    def load_best_model(self):
        study_details = self._get_finetuning_study_details()
        study_name = study_details.pop('study_name')
        optuna_storage = study_details.pop('storage')

        target_dir = optuna_storage.parent / study_name
        target_dir.mkdir(parents=True, exist_ok=True)

        model_name = f"{str(target_dir)}/best_model.bin"
        model = CustomBERTopic.load(path=model_name)
        return model

    def main(self):
        # load key datasets
        self.load_datasets(dataset_index=self.dataset_index, load_test=False)

        study_details = self._get_finetuning_study_details()
        study_name = study_details.pop('study_name')
        storage = study_details.pop("storage")
        optuna_study = optuna.create_study(direction='maximize', storage=f"sqlite:///{str(storage)}",
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

    def lemmatize_sentences(self, sentences, join=True):
        lemmatized_sentences = []

        # TODO just check 0th element, no need to check everything
        sentences = [' '.join(sentence) if isinstance(sentence, list) else sentence for sentence in sentences]

        for doc in self.spacy_nlp.pipe(sentences, batch_size=self.batch_size):
            lemmatized_sentence = list(set([token.lemma_ for token in doc]))

            if join:
                lemmatized_sentence = ' '.join(lemmatized_sentence)

            lemmatized_sentences.append(lemmatized_sentence)

        return lemmatized_sentences

    def calculate_coherence_score(self, model, sentences: List[str]):
        # --------------------------------
        # for calculating coherence score
        cleaned_docs = model._preprocess_text(sentences)
        cleaned_docs = self.drop_sep_tokens(sentences=cleaned_docs)

        # [DONE] do lemmatization on all sentences in df, and just load lemmatized sentences, then clean them using model
        # self.logger.debug(f"Lemmatizing {len(cleaned_docs)} docs")
        # cleaned_docs = self.lemmatize_sentences(sentences=cleaned_docs)
        # self.logger.debug(f"Finished Lemmatizing {len(cleaned_docs)} docs")

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
        self.logger.debug(f"Lemmatizing {len(topic_words)} topic words lists")
        topic_words = self.lemmatize_sentences(sentences=topic_words, join=False)
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
    trainer.main()
    trainer.test()
    # TODO add more cluster evaluation metrics - topic diversity, rand index
