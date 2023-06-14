import re
from logging import Logger

from optuna.trial import Trial


class ClassifierHyperparameters:
    def __init__(self, config: dict, logger: Logger, trial: Trial):
        self.logger = logger
        self.config = config
        self.hyperparameters_config = self.config['hyperparameters']

        # model settings
        self.input_size = self.config.get('input_size', 768)
        self.num_classes = self.config.get('num_classes', 176)

        self.logger.debug(f"Preparing hyperparameters")

        # hyperparameters
        self.model_name = trial.suggest_categorical(name='model_name',
                                                    choices=self.hyperparameters_config['model_name'])

        self.n_epochs = int(
            trial.suggest_categorical(name='n_epochs', choices=self.hyperparameters_config.get('n_epochs', 20)))
        self.learning_rate = float(
            trial.suggest_categorical(name='learning_rate',
                                      choices=self.hyperparameters_config.get('learning_rate', 1.0)))
        self.batch_size = trial.suggest_categorical(name='batch_size',
                                                    choices=self.hyperparameters_config.get('batch_size', 128))

        self.scheduler_step_size = trial.suggest_categorical(name='scheduler_step_size', choices=self.hyperparameters_config.get('scheduler_step_size', 5))
        self.scheduler_gamma = trial.suggest_categorical(name='scheduler_gamma', choices=self.hyperparameters_config.get('scheduler_gamma', 0.01))

        self.classifier_model_name = trial.suggest_categorical(name='classifier_model_name',
                                                               choices=self.hyperparameters_config.get(
                                                                   'classifier_model_name', 'dense'))

        self.logger.debug(f"Finished preparing hyperparameters")

    @property
    def model_normalized_name(self):
        model_normalized_name = re.sub("/", "_", self.model_name)
        return model_normalized_name
