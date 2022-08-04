## The abstract of the research project to which this code belongs:
 The present research investigates the effect of feature enrichment by word vector models for text classification with the BOW representation, on a small and unbalanced dataset. The classification problem involved a binary classification task on a dataset of 424 transcripts that had a label distribution of 0: non-conspiratorial (~80%), and 1: conspiratorial (~20%). The method by which the feature enrichment was executed was by using word vector models with the support vector machine a multi-nominal naïve Bayes classifiers. In total, four word vector models were used for feature enrichment: a  (larger generic) GloVe model trained on Wikipedia 2014 (vocabulary of 400k); and a (smaller domain-specific) self-trained word vector model was trained on a corpus of conspiracy comments between 2015 and 2019 (vocabulary of 213k). All four models that used feature enrichment with the word vector models outperformed the baseline; both the Reddit conspiracy word vector models outperformed the GloVe word vector models. The highest result was by the Reddit conspiracy SVM model, the highest increase in performance relative to its baseline was by the Reddit MNB model.


    
- files without a number are not direclty relevant fort the pipeline (pickle, readme, etc)
 
| __file/step__                    | __State__   | __comments__        |
| -------------                     |:-------------:| ------      |
| 1_Import+Preprocess.py            |done|; for importing transcripts, labels, etc  | 
| 2.0_vectorization_BOW&split.py    |done| for vectorization; | 
| 2.5_Analysis.py                   |done|not relevant for end-result; but I used it for EDA & observing the results of changes in HyperParamaters and feature engineering designs     |
| 3.2_Word2Vec-selfTrained_onRedditConsp.py                 |done| 'the domain specific external corpus/dataset': 40k reddit comments are scraped, from which feature can be extracted (looking into possibility to use googleBigquery to download an even bigger dataset * update 23may: latter was succesfull, now the dataset is 1 MILION comments )
| 3.3_vectorization&FE_Validation.py|done  | Gridsearch for hyperparamaters for vectorization, feature engineering and Classifier processes (ultimately, the hyperparamater set yielding highest validation result): 2 different models are creating from feature engineering(FE) a "general corpus": pretrained external wordVector model (GloVe wikipedia 2014 (400k terms); and a "specific corpus" on reddit, resulting in  ~213K terms|
|4. classifying testtest |done |Script incorperating files before, with a fixed Hyperparamater set (best performing on validation) for testing on new data *update*: we talked about using a different test-set, so test results will likely differ after having done that.  |
