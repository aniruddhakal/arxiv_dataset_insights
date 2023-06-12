import gc

from logging import Logger
import torch
from bertopic._bertopic import *


class CustomBERTopic(BERTopic):
    def __init__(self,
                 logger: Logger,
                 language: str = "english",
                 top_n_words: int = 10,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 10,
                 nr_topics: Union[int, str] = None,
                 low_memory: bool = False,
                 calculate_probabilities: bool = False,
                 seed_topic_list: List[List[str]] = None,
                 embedding_model=None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None,
                 ctfidf_model: TfidfTransformer = None,
                 representation_model: BaseRepresentation = None,
                 verbose: bool = False, ):
        super().__init__(language=language,
                         top_n_words=top_n_words,
                         n_gram_range=n_gram_range,
                         min_topic_size=min_topic_size,
                         nr_topics=nr_topics,
                         low_memory=low_memory,
                         calculate_probabilities=calculate_probabilities,
                         seed_topic_list=seed_topic_list,
                         embedding_model=embedding_model,
                         umap_model=umap_model,
                         hdbscan_model=hdbscan_model,
                         vectorizer_model=vectorizer_model,
                         ctfidf_model=ctfidf_model,
                         representation_model=representation_model,
                         verbose=verbose)
        self.logger = logger

    def fit_transform(self,
                      documents: List[str],
                      embeddings: np.ndarray = None,
                      images: List[str] = None,
                      y: Union[List[int], np.ndarray] = None) -> Tuple[List[int],
    Union[np.ndarray, None]]:
        """ Overriding parent class method
        """
        self.logger.debug(f"Calling overriden fit_transform method")
        if documents is not None:
            check_documents_type(documents)
            check_embeddings_shape(embeddings, documents)

        doc_ids = range(len(documents)) if documents is not None else range(len(images))
        documents = pd.DataFrame({"Document": documents,
                                  "ID": doc_ids,
                                  "Topic": None,
                                  "Image": images})

        # Extract embeddings
        if embeddings is None:
            self.embedding_model = select_backend(self.embedding_model,
                                                  language=self.language)
            embeddings = self._extract_embeddings(documents.Document.values.tolist(),
                                                  images=images,
                                                  method="document",
                                                  verbose=self.verbose)
            logger.info("Transformed documents to Embeddings")
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)

        # Reduce dimensionality
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)
        umap_embeddings = self._reduce_dimensionality(embeddings, y)
        self.logger.debug(f"Finished dimensionality reduction, cleaning up memory")
        torch.cuda.empty_cache()
        gc.collect()

        # Cluster reduced embeddings
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)
        self.logger.debug(f"Finished clustering embeddings, cleaning up memory")
        del umap_embeddings
        torch.cuda.empty_cache()
        gc.collect()

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Create documents from images if we have images only
        if documents.Document.values[0] is None:
            custom_documents = self._images_to_text(documents, embeddings)

            # Extract topics by calculating c-TF-IDF
            self._extract_topics(custom_documents, embeddings=embeddings)
            self._create_topic_vectors(documents=documents, embeddings=embeddings)

            # Reduce topics
            if self.nr_topics:
                custom_documents = self._reduce_topics(custom_documents)

            # Save the top 3 most representative documents per topic
            self._save_representative_docs(custom_documents)
        else:
            # Extract topics by calculating c-TF-IDF
            self._extract_topics(documents, embeddings=embeddings)

            # Reduce topics
            if self.nr_topics:
                documents = self._reduce_topics(documents)

            # Save the top 3 most representative documents per topic
            self._save_representative_docs(documents)

        # Resulting output
        self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        self.logger.debug(f"Finished fit_transform call, cleaning up memory")
        torch.cuda.empty_cache()
        gc.collect()

        return predictions, self.probabilities_

    def partial_fit(self,
                    documents: List[str],
                    embeddings: np.ndarray = None,
                    y: Union[List[int], np.ndarray] = None):
        """
        Overriding parent class method
        """

        self.logger.debug(f"Calling overriden partial_fit method")
        # Checks
        check_embeddings_shape(embeddings, documents)
        if not hasattr(self.hdbscan_model, "partial_fit"):
            raise ValueError("In order to use `.partial_fit`, the cluster model should have "
                             "a `.partial_fit` function.")

        # Prepare documents
        if isinstance(documents, str):
            documents = [documents]
        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        if embeddings is None:
            if self.topic_representations_ is None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)
            embeddings = self._extract_embeddings(documents.Document.values.tolist(),
                                                  method="document",
                                                  verbose=self.verbose)
        else:
            if self.embedding_model is not None and self.topic_representations_ is None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)

        # Reduce dimensionality
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)
        umap_embeddings = self._reduce_dimensionality(embeddings, y, partial_fit=True)
        self.logger.debug(f"Finished dimensionality reduction, cleaning up memory")
        torch.cuda.empty_cache()
        gc.collect()

        # Cluster reduced embeddings
        documents, self.probabilities_ = self._cluster_embeddings(umap_embeddings, documents, partial_fit=True)
        topics = documents.Topic.to_list()
        self.logger.debug(f"Finished clustering embeddings, cleaning up memory")
        del umap_embeddings
        torch.cuda.empty_cache()
        gc.collect()

        # Map and find new topics
        if not self.topic_mapper_:
            self.topic_mapper_ = TopicMapper(topics)
        mappings = self.topic_mapper_.get_mappings()
        new_topics = set(topics).difference(set(mappings.keys()))
        new_topic_ids = {topic: max(mappings.values()) + index + 1 for index, topic in enumerate(new_topics)}
        self.topic_mapper_.add_new_topics(new_topic_ids)
        updated_mappings = self.topic_mapper_.get_mappings()
        updated_topics = [updated_mappings[topic] for topic in topics]
        documents["Topic"] = updated_topics

        # Add missing topics (topics that were originally created but are now missing)
        if self.topic_representations_:
            missing_topics = set(self.topic_representations_.keys()).difference(set(updated_topics))
            for missing_topic in missing_topics:
                documents.loc[len(documents), :] = [" ", len(documents), missing_topic]
        else:
            missing_topics = {}

        # Prepare documents
        documents_per_topic = documents.sort_values("Topic").groupby(['Topic'], as_index=False)
        updated_topics = documents_per_topic.first().Topic.astype(int)
        documents_per_topic = documents_per_topic.agg({'Document': ' '.join})

        # Update topic representations
        self.c_tf_idf_, updated_words = self._c_tf_idf(documents_per_topic, partial_fit=True)
        self.topic_representations_ = self._extract_words_per_topic(updated_words, documents, self.c_tf_idf_, calculate_aspects=False)
        self._create_topic_vectors()
        self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in self.topic_representations_.items()}

        # Update topic sizes
        if len(missing_topics) > 0:
            documents = documents.iloc[:-len(missing_topics)]

        if self.topic_sizes_ is None:
            self._update_topic_size(documents)
        else:
            sizes = documents.groupby(['Topic'], as_index=False).count()
            for _, row in sizes.iterrows():
                topic = int(row.Topic)
                if self.topic_sizes_.get(topic) is not None and topic not in missing_topics:
                    self.topic_sizes_[topic] += int(row.Document)
                elif self.topic_sizes_.get(topic) is None:
                    self.topic_sizes_[topic] = int(row.Document)
            self.topics_ = documents.Topic.astype(int).tolist()

        self.logger.debug(f"Finished fit_transform call, cleaning up memory")
        torch.cuda.empty_cache()
        gc.collect()

        return self
