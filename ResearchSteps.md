- When choosing models please use TensorFlow or Keras based.

### Deliverable

- A GitHub repository with production ready code and a comprehensive README
- [Optional] Data visualization to communicate your analysis

## Use abstracts for classification and subject categories for class association.

## My Steps

### Literature Review

- #### TODO Report top 3 articles referencing the im implementation you will be using
- #### A Sentence-level Hierarchical BERT Model for Document Classification with Limited Labelled Data (https://arxiv.org/pdf/2106.06738.pdf)
    - Hierarchical BERT to extract document representation of whole abstract
- #### BERTopic: Neural topic modeling with a class-based TF-IDF procedure (https://arxiv.org/abs/2203.05794)
- #### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (https://arxiv.org/pdf/1908.10084.pdf)
  - https://github.com/UKPLab/sentence-transformers
- #### Influence of various text embeddings on clustering performance in NLP (https://arxiv.org/pdf/2305.03144.pdf)
  - Extract CLS (intended for classification purposes), Average (pooler) state and apply clustering on it

- #### Label Encoding

### EDA

    - TODO Perform exploratory data analysis to understand the structure and characteristics of the dataset.
	- average sentences (512 token multiples) per abstract
	- average number of words per abstract
    - explore and do EDA based on certain fields

### Preprocess the data

The input text is the abstract of scientific papers. So, I'd assume use of stop words serves meaningful purpose, and
therefore shouldn't be removed.  
During the data exploration, the essential steps to the need for preprocessing will revolve around finding any
characters that aren't supposed
to be the part of usual abstracts. e.g. Some symbols that might be encoded incorrectly into the text, or finding some
out of vocabulary (OOV) words, and characters.
The WordPiece tokenizer of BERT would still make sure to assign appropriate token to almost every possible word in the
corpus, therefore, there might not be any need for us to handle OOV words explicitly.

- TODO check if there are any words missing from the original text after the tokenizer generated text is decoded back to
  original text.
    - If 100% of the original text is restored, or if the missing portion isn't significant enough in terms of the
      contribution towards the domain knowledge,
    - we don't need to worry about handling any OOV words / tokens explicitly.

### Train classifier
#### - Model choice steps
    - Filter bert models for "Feature Extraction" category
    - Model size
    - Inference time
    - Number of attention heads
    - Arhcitecture (e.g. SBERT)
    - Specific downstream tasks / datasets trained / finetuned on
    - Domain specific models suitable for arxiv dataset feature extraction
    - Model's documentation - model card, performance, intended use, limitations or considerations
    - Experiment and evaluate

    - a suitable NLP model for text classification
        - (use hugging face for that
        - https://huggingface.co/models?library=tf&language=en&license=license:apache-2.0&sort=downloads&search=bert-base )

### Evaluate the model's performance on the validation set using appropriate metrics.

### Perform hyperparameter tuning or model selection to improve the model's performance.

### Conclude your findings in a final report When choosing models

    - please use TensorFlow or Keras based.

### Data visualization to communicate your analysis

### Issues faced
- Installation related issues
- Issue with mounting GPU to docker container [unresolved]
- Issues with extracting Embeddings for Clustering, the model, size, and time limit considerations, computations limits on local machine
- Issues while installing cuML library
- Issues faced while trying evaluation metrics from OCTIS
