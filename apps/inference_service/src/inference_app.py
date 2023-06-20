import numpy as np
import pickle
from sklearn.preprocessing import Binarizer
from abstract_classification_models import ArxivAbstractClassifier
from util import *
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Union
from logging import Logger
# from custom_bertopic import CustomBERTopic

from flask import Flask, request

app = Flask(__name__)


class ArxivInferenceApp:
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.dataset_path = Path(config['dataset_path'])
        self.models_path = Path(config['models_path'])

        # model settings
        self.transformer_model_name = config['transformer_model_name']
        self.classifier_study_name = config['classifier_study_name']
        self.topic_modeling_study_name = config['topic_modeling_study_name']
        self.label_transformer_file = config['label_transformer_filename']

        self.num_classes = config.get('num_classes', 176)
        self.input_size = config.get('input_size', 768)

        self.classifier_threshold = config.get('classifier_threshold', 0.1)

        self.binarizer = Binarizer(threshold=self.classifier_threshold)
        self.sbert_model = None
        self.bertopic_model = None
        self.multilabel_binarizer = None
        self.classifier_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_classes(self):
        abstract_text = request.args.get('abstract')
        self._load_classifier_model()
        embedding = self._extract_transformer_embedding(abstract_text=abstract_text)
        embedding = torch.tensor(embedding, device=self.device)
        outputs = self.classifier_model(embedding)
        binary_preds = self.binarizer.transform(outputs.detach().cpu().numpy())
        self.logger.debug(f"Inference done, predicting labels")
        predicted_labels = self.multilabel_binarizer.inverse_transform(binary_preds)

        self.logger.debug(f"Processed successfully")

        return {
            "abstract": abstract_text,
            "predicted_labels": predicted_labels
        }

    # def get_topics(self):
    #     # extract abstract text from payload
    #     abstract_text = request.args.get('abstract')
    #
    #     # load bertopic mdoel
    #     return "Not supported yet"

    def _load_classifier_model(self):
        if self.classifier_model is None:
            self.classifier_model = ArxivAbstractClassifier(
                input_size=self.input_size,
                num_classes=self.num_classes)
            model_file = self.models_path / 'finetuning_studies' / self.classifier_study_name / 'best_model.bin'
            self.classifier_model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
            self.classifier_model = self.classifier_model.to(self.device)

        if self.multilabel_binarizer is None:
            self.logger.debug(f"Loading multilabel binarizer")
            with open(self.models_path / self.label_transformer_file, 'rb') as f:
                self.multilabel_binarizer = pickle.load(f)
                self.logger.debug(f"Multilabel binarizer loaded, classes supported: {len(self.multilabel_binarizer.classes_)}")

    # def _load_bertopic_model(self):
    #     if self.bertopic_model is None:
    #         self.logger.debug(f"Loading Best BERTopic model from study {self.topic_modeling_study_name}")
    #         self.bertopic_model = CustomBERTopic.load(
    #             self.models_path / 'finetuning_studies' / self.topic_modeling_study_name / 'best_model.bin')

    def _extract_transformer_embedding(self, abstract_text: str) -> Union[np.ndarray, None]:
        self.load_transformer_model()

        embeddings = None

        if isinstance(abstract_text, str) and len(abstract_text.strip()) > 0:
            self.logger.debug(f"Extracting embeddings for abstract of length {len(abstract_text)}")

            embeddings = self.sbert_model.encode(sentences=[abstract_text],
                                                 batch_size=1,
                                                 device=self.device,
                                                 convert_to_numpy=True,
                                                 show_progress_bar=True)

        return embeddings

    def load_transformer_model(self):
        if self.sbert_model is None:
            self.logger.debug(f"Loading transformer model {self.transformer_model_name}")
            self.sbert_model = SentenceTransformer(model_name_or_path=self.transformer_model_name,
                                                   device=self.device)


if __name__ == '__main__':
    args = parse_arguments()
    config = load_config(args.config_file)

    logfile_name = get_logfile_name(config=config)
    logger = initialize_logger(logfile_name=logfile_name, log_level=config['logging_level'])

    inference_app = ArxivInferenceApp(config=config, logger=logger)

    app.add_url_rule("/class_for_abstract", view_func=inference_app.get_classes)
    app.add_url_rule("/topics_for_abstract", view_func=inference_app.get_classes)

    app.run(host=config['app_host'], port=config['app_port'], debug=True)
