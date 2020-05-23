[link to markupt help!](https://guides.github.com/features/mastering-markdown/)

# Conspiracy_Siebe_Project_Specific

update: may 23)
- Highest macro-f1 on validation:  ; 0.82 (GloVe model) vs 0.48 baseline
- highest macro-f1 on test-set (self watched videos) ; 0.63 (Glove model) vs 0.48 baseline

**__TBD:__**
...

__Info on the files within the repository:__
  - the files that are numbered are part of the text classification pipeline; their respective number is their place within that order. 
    - the files startin with 'FE'='feature_Enrichment are the core of the feature_engineering process; eventually, their output models will also be input in the TextPipeline from step 3
    
- files without a number are not part of the pipeline (pickle, readme, etc)
  
 
| __file/step__                    | __State__   | __comments__        |
| -------------                     |:-------------:| ------      |
| 1_Import+Preprocess.py            |done|; for importing transcripts, labels, etc  | 
| 2.0_vectorization_BOW&split.py    |done| for vectorization; | 
| 2.5_Analysis.py                   |done|not relevant for end-result; but I used it for EDA & observing the results of changes in HyperParamaters and feature engineering designs     |
| FE1_RedditCorpus                  |done| 'the domain specific external corpus/dataset': 40k reddit comments are scraped, from which feature can be extracted (looking into possibility to use googleBigquery to download an even bigger dataset * update 23may: latter was succesfull, now the dataset is 1 MILION comments )
| 3.3_vectorization&FE_Validation.py|done  | Gridsearch for hyperparamaters for vectorization, feature engineering and Classifier processes (ultimately, the hyperparamater set yielding highest validation result): 2 different models are creating from feature engineering(FE) a "general corpus": pretrained external wordVector model (GloVe wikipedia 2014 (400k terms); and a "specific corpus" on reddit, resulting in  ~213K terms|
