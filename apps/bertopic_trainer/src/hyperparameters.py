import re
from logging import Logger

from optuna.trial import Trial


class BertopicHyperparameters:
    def __init__(self, config: dict, trial: Trial, logger: Logger):
        self.logger = logger
        self.config = config
        self.hyperparameters_config = self.config['hyperparameters']

        self.logger.debug(f"Preparing hyperparameters")

        # hyperparameters
        self.model_name = trial.suggest_categorical(name='model_name',
                                                    choices=self.hyperparameters_config['model_name'])

        self.nr_topics = int(
            trial.suggest_categorical(name='nr_topics', choices=self.hyperparameters_config.get('nr_topics', 30)))
        self.top_n_words = int(trial.suggest_categorical(name='top_n_words',
                                                         choices=self.hyperparameters_config.get('top_n_words', 100)))
        self.min_topic_size = int(trial.suggest_categorical(name='min_topic_size',
                                                            choices=self.hyperparameters_config.get('min_topic_size',
                                                                                                    10)))
        self.n_gram_range_start = int(trial.suggest_categorical(name='n_gram_range_start',
                                      choices=self.hyperparameters_config.get('n_gram_range_start', 1)))
        self.n_gram_range_end = int(trial.suggest_categorical(name='n_gram_range_end',
                                                                choices=self.hyperparameters_config.get(
                                                                    'n_gram_range_end', 1)))

        # count vectorizer params
        self.max_features = int(trial.suggest_categorical(name='max_features',
                                                          choices=self.hyperparameters_config.get('max_features', 100)))
        self.max_df = float(trial.suggest_categorical(name='max_df',
                                                      choices=self.hyperparameters_config.get('max_df', 0.8)))
        self.min_df = float(trial.suggest_categorical(name='min_df',
                                                      choices=self.hyperparameters_config.get('min_df', 0.05)))
        self.lowercase = trial.suggest_categorical(name='lowercase',
                                                   choices=self.hyperparameters_config.get('lowercase', True))

        # umap params
        self.n_neighbors = int(
            trial.suggest_categorical(name='n_neighbors', choices=self.hyperparameters_config.get('n_neighbors', 15)))
        self.n_components = int(
            trial.suggest_categorical(name='n_components', choices=self.hyperparameters_config.get('n_components', 50)))
        self.umap_metric = trial.suggest_categorical(name='umap_metric',
                                                     choices=self.hyperparameters_config.get('umap_metric',
                                                                                             'euclidean'))
        self.n_epochs = int(
            trial.suggest_categorical(name='n_epochs', choices=self.hyperparameters_config.get('n_epochs', 200)))
        self.learning_rate = int(
            trial.suggest_categorical(name='learning_rate',
                                      choices=self.hyperparameters_config.get('learning_rate', 1.0)))
        self.min_dist = int(
            trial.suggest_categorical(name='min_dist', choices=self.hyperparameters_config.get('min_dist', 0.1)))
        self.random_state = int(
            trial.suggest_categorical(name='random_state', choices=self.hyperparameters_config.get('random_state', 65)))

        # hdbscan params
        self.min_cluster_size = int(
            trial.suggest_categorical(name='min_cluster_size',
                                      choices=self.hyperparameters_config.get('min_cluster_size', 5)))
        self.cluster_selection_epsilon = int(
            trial.suggest_categorical(name='cluster_selection_epsilon',
                                      choices=self.hyperparameters_config.get('cluster_selection_epsilon', 0.1)))
        self.hdbscan_metric = trial.suggest_categorical(name='hdbscan_metric',
                                                        choices=self.hyperparameters_config.get('hdbscan_metric',
                                                                                                'euclidean'))
        self.cluster_selection_method = trial.suggest_categorical(name='cluster_selection_method',
                                                                  choices=self.hyperparameters_config.get(
                                                                      'cluster_selection_method', 'eom'))

        # metrics params
        self.topk = int(trial.suggest_categorical(name='topk',
                                                  choices=self.hyperparameters_config.get('topk', 10)))

        self.logger.debug(f"Finished preparing hyperparameters")

    @property
    def model_normalized_name(self):
        model_normalized_name = re.sub("/", "_", self.model_name)
        return model_normalized_name
