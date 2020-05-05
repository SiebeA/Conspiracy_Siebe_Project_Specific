# Conspiracy_Siebe_Project_Specific

This is the repository that is specific to Siebe's project

Info on the files within the repository:
  - the files that are numbered are subject to be executed within an order i.e. my text classification pipeline; their respective number is their place within that order. Here, the baseline will be set, after the model being optimized in 3Gridsearch(preprocess-vectorizationHyperparametersettings-ClassifierHyperparamaterSettings) 
  
  - the files under the 'readme file' (with 'featureEnrichment' in the title) are for feature_engineering; eventually, these models will also be input in the TextPipeline from step 3 (GridsearchCV) onwards: then the results can be compared and the Research Question--Can the use of external data improve the performance of a text classification model for detecting conspiratorial video transcripts on a small unbalanced dataset?--can be answered, as the models includes externally derived features, by which the original features are enriched. 
  
  | Tables        | done?           |
| ------------- |:-------------:|
| 1_Import+Preprocess.py     |yes  | 
| 2.0_vectorization_BOW&split.py    |Gridsearch-system=setup: Hyperparamaters can be optimzed even more  | 
| 2.5_Analysis.py| not relevant     |
| 3_GridsearchCV.py  | Gridsearch-system=setup: Hyperparamaters of baseline classifier can be optimzed even more  |
| FE1_RedditCorpus    | 40k reddit comments are scraped, from which feature can be extracted (looking into possibility to use googleBigquery to download an even bigger dataset)
| FE2_FeatureEnrichment  | Working on it; vectorization&classifier Hyperparamaters (can(and ought!) be easily extended here|
