from gensim.parsing.preprocessing import STOPWORDS
from logging import Logger


class BertopicHyperparameters:
    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.logger = logger
        self.hyperparameters_config = self.config['hyperparameters']

        self.logger.debug(f"Preparing hyperparameters")
        # hyperparameters
        self.nr_topics = self.hyperparameters_config.get('nr_topics', 30)
        self.top_n_words = self.hyperparameters_config.get('top_n_words', 100)
        self.min_topic_size = self.hyperparameters_config.get('min_topic_size', 10)
        self.n_gram_range = self.hyperparameters_config.get('n_gram_range', (1, 1))

        # TODO inputs for hyperparameters
        self.min_categories = self.hyperparameters_config.get('min_categories', 5)
        self.max_categories = self.hyperparameters_config.get('max_categories', 5)

        # count vectorizer params
        self.max_features = self.hyperparameters_config.get('max_features', 100)
        self.max_df = self.hyperparameters_config.get('max_df', 0.8)
        self.min_df = self.hyperparameters_config.get('min_df', 0.05)
        self.lowercase = self.hyperparameters_config.get('lowercase', True)
        self.stop_words = self.hyperparameters_config.get('stop_words', STOPWORDS)

        # metrics params
        self.topk = self.hyperparameters_config.get('topk', 10)

        self.logger.debug(f"Finished preparing hyperparameters")

    def suggest_trial(self):
        # TODO Implement
        #   suggest (set to self) optuna trial hyperparameters,
        #   while, also registering the details within otuna trial
        raise NotImplementedError()

    def capture_best_hyperparameters(self):
        raise NotImplementedError()