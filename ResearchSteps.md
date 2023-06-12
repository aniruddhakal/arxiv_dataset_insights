- When choosing models please use TensorFlow or Keras based.

### Deliverable

- A GitHub repository with production ready code and a comprehensive README
- [Optional] Data visualization to communicate your analysis

## Use abstracts for classification and subject categories for class association.

## Research Steps
1. [Literature Review](#literature-review)
  1. [Top Articles / Papers / Implementations Referred](#top-articles--papers--implementations-referred)
1. [Exploratory Data Analysis](#eda)
1. [Data Preprocessing](#data-preprocessing)
1. [Train a Classifier](#train-a-classifier)
1. [Performance Evaluations](#performance-evaluation-)
1. [Hyperparameter Tuning](#hyperparameter-tuning-)
1. [Conclusions](#conclusions)
1. [Data Visualizations](#data-visualizations)
1. [Issues Faced](#issues-faced)

## Literature Review
  - Classifying documents into unknown number of classes and figuring out the optimal number of classes is majorly
  solved using these approaches:
    1. Clustering
    2. Topic Modeling
  - The issue with Clustering based approaches is it doesn't take document topic prior and word topic prior, which is
  addressed by topic modeling approaches such as LDA.
  - While Topic Modeling algorithms such as LDA account for word and document topic prior's, they still rely on the traditional
  CountVectorizer, or TF-IDF based approaches yet using the smaller dictionary with limited vocabulary. The preprocessing logic
  thus relies on removing stop-words, followed by either stemming or lemmatization for the remaining tokens, so that most of the similar
  tokens, irregardless of the context, are accounted for. So that having brought all words to their lexical base forms, they contribute
  to the overall term frequency as the same word.
  - Although, we are still giving up on contextual meaning of word by just stemming or lemmatizing them. Which can be overcome by utilizing
  the modern Deep Learning architectures such as Transformers based models. Transformers allow us to extract mainly two types of representations:
    1. Token Level Representation - Different embedding for every word (depending on surrounding context).
    2. Document Level Representation - Pooling layer representation (average of all word representations for a given document).
  - We can either go heavy and try clustering individual token level representations for all documents. Which is unexplored in our current study.
  - Or we can simply go with option 2 - Clustering Document Level Representation.
  - Either way, we'd still end up in a challenge of selecting top words for topics, or having too many topics, and face challenge of reducing / merging
  similar topics.
  - Saha et. al. in [4] has demonstrated use of KMeans, DBSCAN, and HDBSCAN clustering algorithms on BERT-Average, BERT-CLS, and Word2Vec embeddings
  which is evaluated across metrics such as Silhouette Score, Adjusted Rand Index Score, Purity Score etc.
  - The study demonstrates that BERT embeddings mostly best the Word2Vec embeddings across most of the
  clustering techniques and evaluation metrics.
  - Although, their study is fairly limited to just single dataset comprised of consumer electronics from one brand.
  - Further to that, it also becomes crucial on which model do we use to extract embeddings. Jinghui et. al. in [2] introduced
  SBERT where they build Siamese network on top of Transformer models, and finetune the network for sentence similarity task,
  making it much more faster and easier to find similar sentence pairs.
  - Although, we still face computational challenges trying to extract embeddings from SBERT model variants given the size of the arxiv dataset,
  hence, we go further exploring options to reduce number of parameters used by the model. Which brings us to Reimers et. al. [3],
  where they further expand SBERT models by doing the following:
    1. Produce Distilled Models (significantly reduced number of parameters)
    2. Finetune models on Multiple Languages to bring same words from multiple languages in the same vector space. 
  - Additionally, for model selection, we select the following 4 models for hyperparameter tuning, among which, we can endorse
  the findings of Reimers et. al. where they say - "Even though SBERT-nli-stsb was trained on the STSbenchmark train set,
  we observe the best performance by SBERT-paraphrase, which was not trained with any STS dataset.
  Instead, it was trained on a large and broad paraphrase corpus, mainly derived from Wikipedia, which generalizes well to various topics."
    1. `sentence-transformers/distilroberta-base-paraphrase-v1`
    2. `sentence-transformers/stsb-distilroberta-base-v2`
    3. `sentence-transformers/distilbert-base-nli-stsb-quora-ranking`
    4. `distilbert-base-nli-mean-tokens`
  - As a result of our hyperparameter tuning for model selection, `sentence-transformers/distilroberta-base-paraphrase-v1` indeed
  topped our list, and hence, the further hyperparameter tuning was done using the same model.

### Top articles / papers / implementations referred:
- #### 1. BERTopic: Neural topic modeling with a class-based TF-IDF procedure (https://arxiv.org/abs/2203.05794)
  - Library Documentation - https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html
- #### 2. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (https://arxiv.org/pdf/1908.10084.pdf)
  - Library Documentation - https://github.com/UKPLab/sentence-transformers
- #### 3. Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation (https://arxiv.org/abs/2004.09813)
- #### 4. Influence of various text embeddings on clustering performance in NLP (https://arxiv.org/pdf/2305.03144.pdf)
  

## EDA

    - TODO Perform exploratory data analysis to understand the structure and characteristics of the dataset.
	- average sentences (512 token multiples) per abstract
	- average number of words per abstract
    - explore and do EDA based on certain fields

## Data Preprocessing

- The input text is the abstract of scientific papers. So, I'd assume use of stop words serves meaningful purpose, and
therefore shouldn't be removed before inference from transformer models. Although, I remove stop words when CountVectorizer is called.
And the list of stopwords is used from Gensim library - `gensim.parsing.preprocessing.STOPWORDS`

- Basic cleaning - I only performed basic text cleaning as follows:
  - Replace `\n`'s with a space.
  - Add `[SEP]` token when sentence is finished with full stop, because pretrained tokenizers don't do it themselves,
  and hence, always miss the special token `[SEP]`, which has impact on the final representation of the document.
    - Although, in initial experiment, the downstream model I ended up using is `sentence-transformers/distilroberta-base-paraphrase-v1`.
    - And since `RoBERTa` model rather uses `<sep>` as a separator token, and my preprocessed data still contains `[SEP]` tokens, `sep` is going
    to exist across all the documents as a noise.
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

## Train a Classifier
### - Model choice steps
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

## Performance Evaluation 
- Evaluate the model's performance on the validation set using appropriate metrics.

## Hyperparameter Tuning 
- Perform hyperparameter tuning or model selection to improve the model's performance.
![img.png](resources/img.png)
The image above shows `cluster_selection_eps of 0.2`, and `min_cluster_size of 200, and 600` contributing to `higher coherence score of 0.6 and above`.

## Conclusions
- Conclude your findings in a final report When choosing models

    - please use TensorFlow or Keras based.

## Data Visualizations

## Issues faced
- Installation related issues
- Issue with mounting GPU to docker container [unresolved]
- Issues with extracting Embeddings for Clustering, the model, size, and time limit considerations, computations limits on local machine
- Issues while installing cuML library
- Issues faced while trying evaluation metrics from OCTIS
- GPU memory leaks and inability to proceed with finetuning of larger datasets we selected - datasets 1, 2, 3, 4
  - Although I did manage to train 1 model on dataset 1, for which, the training data is approx 40% of the original datset.
    - That too was achieved by only reducing precision of embeddings to float16.
    - Clearing GPU context memory after intermediate steps of dimensionality reduction, and clustering, but not as much for bigger datasets.
- Difficulty in reproducing BERTopic results, at least in our case, due to using GPU based computations across 80% processing. 