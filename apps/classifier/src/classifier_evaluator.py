from sklearn.preprocessing import Binarizer

from abstract_classifier_trainer import *
from arxiv_dataset import ArxivDataset


class DummyHyperparameters:
    def __init__(self, model_name: str, input_size: int, num_classes: int):
        self.model_name = model_name
        self.input_size = input_size
        self.num_classes = num_classes

    @property
    def model_normalized_name(self):
        model_normalized_name = re.sub("/", "_", self.model_name)
        return model_normalized_name


class Evaluator:
    def __init__(self, logger: Logger, config: Dict, trainer: AbstractClassificationTrainer):
        self.logger = logger
        self.config = config

        self.trainer = trainer

        self.dataset_path = Path(self.config['dataset_path'])
        self.models_path = Path(self.config['models_path'])

        self.study_name = self.config['optuna_study_name']
        self.dataset_index = self.config['dataset_index']

        self.batch_size = self.config.get('batch_size', 1024)

        self.label_transformer_file = self.models_path / self.config['label_transformer']
        self.model_file = self.models_path / 'finetuning_studies' / self.study_name / 'best_model.bin'

        self.transformer_model_name = self.config['transformer_model_name']
        self.dummy_hyperparameters = DummyHyperparameters(model_name=self.transformer_model_name,
                                                          num_classes=self.config.get('num_classes'),
                                                          input_size=self.config.get('input_size'))

        self.trainer.hyperparameters = self.dummy_hyperparameters

    def inference_loop(self, model: nn.Module, dataloader: DataLoader):
        model.eval()
        validation_loss = 0.0

        predictions = []
        val_loop_y = []

        criterion = nn.BCELoss()
        criterion = self.trainer.move_to_device(criterion)

        for inputs, labels in tqdm(dataloader):
            val_loop_y.extend(labels.tolist())
            with torch.no_grad():
                inputs = self.trainer.move_to_device(inputs)
                labels = self.trainer.move_to_device(labels)

                # Forward pass
                outputs = model(inputs)
                predictions.extend(outputs.tolist())

                # Compute the loss
                loss = criterion(outputs, labels)
                # predictions.extend(outputs.tolist())

                validation_loss += loss.item()

        validation_loss = validation_loss / len(dataloader)
        return validation_loss, predictions, val_loop_y

    def load_best_model(self):
        model = ArxivAbstractClassifier(
            input_size=self.dummy_hyperparameters.input_size,
            num_classes=self.dummy_hyperparameters.num_classes)
        model.load_state_dict(torch.load(self.model_file, map_location=torch.device("cpu")))
        return model

    def evaluate_split(self, split: str):
        self.logger.info(f"Running evaluations for {split} split")

        split_embeddings = self.trainer.extract_or_load_embeddings(split_name=split)

        dataset = self.trainer.load_dataset(split=split, cache=False, dataset_index=self.dataset_index)
        labels =self.trainer.transform_labels(dataset['categories_list'])
        self.logger.debug(f"Loaded dataset labels")

        del dataset
        gc.collect()

        arxiv_dataset = ArxivDataset(data=split_embeddings, targets=labels.astype(np.float32))
        dataloader = DataLoader(dataset=arxiv_dataset, batch_size=self.batch_size, num_workers=1)

        # load model
        model = self.load_best_model()
        model = self.trainer.move_to_device(model)
        self.logger.debug(f"Loaded best model")

        loss, predictions, loop_y = self.inference_loop(model=model, dataloader=dataloader)

        self.logger.debug(f"Running evaluations")
        scores_dict = self.trainer.run_evaluations(split=split, labels=loop_y, predictions=predictions)

        self.logger.info(f"Scores for {split} split: {scores_dict}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    def top_n_accuracy(self, labels, predictions, range_start: int=None, range_end: int=None, threshold=0.1, top=1):
        if range_start == None:
            range_start = 0

        if range_end == None:
            range_end = len(labels)

        binarizer = Binarizer(threshold=threshold)

        binary_preds = binarizer.transform(predictions)

        matches = np.logical_and(labels[range_start:range_end], binary_preds[range_start:range_end])

        # Calculate the accuracy
        accuracy = np.mean(np.any(matches, axis=1))
        logger.info(f"Top {top} accuracy: {accuracy} @ threshold {threshold}")

    def main(self):
        # load key datasets
        self.trainer.load_datasets(dataset_index=self.dataset_index, load_test=False)
        self.evaluate_split('train')
        self.evaluate_split('test')
        self.evaluate_split('validation')


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(args.config_file)

    logfile_name = get_logfile_name(config=config)

    logger = initialize_logger(logfile_name=logfile_name, log_level=config['logging_level'])
    trainer = AbstractClassificationTrainer(config=config, logger=logger)
    evaluator = Evaluator(config=config, logger=logger, trainer=trainer)
    evaluator.main()
