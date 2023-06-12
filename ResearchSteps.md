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

### Data Preprocessing

- The input text is the abstract of scientific papers. So, I'd assume use of stop words serves meaningful purpose, and
therefore shouldn't be removed before inference from transformer models.

- Basic cleaning - I only performed basic text cleaning as follows:
  - Replace `\n`'s with a space.
  - Add `[SEP]` token when sentence is finished with full stop, because pretrained tokenizers don't do it themselves,
  and hence, always miss the special token `[SEP]`, which has impact on the final representation of the document.
    - Although, in initial experiment, the downstream model I ended up using is `sentence-transformers/distilroberta-base-paraphrase-v1`.
    - And since `RoBERTa` model rather uses `<sep>` as a separator token, and my preprocessed data still contains `[SEP]` tokens, `sep` is going
    to exist across all the topics as a noise.
    - Hence, there's a risk of having a lot of similarity across all the documents just because of this mistake.
    - **[TODO for myself]** I'll fix this in coming iterations. I'm just choosing to continue as is because I have already
    generated data splits, and corresponding document embeddings for abstracts.
    and I don't want to spend anymore time re-doing all that in initial iteration.

- Because I have chosen to work with BERTopic, which, after extracting
embeddings from transformer models, the only place where original text is then used is in c-tfidf model.
Since, c-tfidf model still uses traditional CountVectorizer and TF-IDF Transformer on top of it, where contextual word embeddings from
transformer model aren't used to compute similarities. And hence, it's necessary to perform lemmatization on all documents, so that words with similar base lexical meaning aren't treated differently.
In this implementation, I only performed offline lemmatization and ensure they are used only after the for c-tf-idf step, while the transformer embeddings are generated
using the full abstract text with basic cleaning as covered in previous step.

- The WordPiece tokenizer of BERT would still make sure to assign appropriate token to almost every possible token in the
corpus, therefore, there might is no need for us to handle OOV words explicitly. Except, where I started seeing some irrelevant words / symbols 
making up the part of the topic labels. Then I can consider removing such symbols / tokens.

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
![img.png](resources/img.png)
The image above shows `cluster_selection_eps of 0.2`, and `min_cluster_size of 200, and 600` contributing to `higher coherence score of 0.6 and above`.

### Conclude your findings in a final report When choosing models

    - please use TensorFlow or Keras based.

### Data visualization to communicate your analysis

### Issues faced
- Installation related issues
- Issue with mounting GPU to docker container [unresolved]
- Issues with extracting Embeddings for Clustering, the model, size, and time limit considerations, computations limits on local machine
- Issues while installing cuML library
- Issues faced while trying evaluation metrics from OCTIS
