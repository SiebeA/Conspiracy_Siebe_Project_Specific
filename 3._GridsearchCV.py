# =============================================================================
# # pickle here, only x,y
# =============================================================================


#======================================================================== #
'  split train test '
#======================================================================== #
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2, random_state=4, shuffle=True, stratify=Y)

#======================================================================== #
' gridsearch         '
#======================================================================== #
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


mnb_pipeline = Pipeline(
        [('vectorizer', CountVectorizer()),
         ('mnb', MultinomialNB()) ]
        )
# I NEED TO RUN THE MY_CLEANER IN 2.0
param_grid =  {
        'vectorizer__max_features':list(range(1000,25000,5000)),#til 25k steps of 5k
        'vectorizer__ngram_range': [(1, 1)],#, (1, 2)],
        'vectorizer__tokenizer': (None, my_cleaner_noLemma_noStop),
        'mnb__alpha':np.linspace(1e-4, 1e0, 2)   #[round( x * 0.1, 2) for x in range(1, 6)]
                }

gs_mnb = GridSearchCV(mnb_pipeline, param_grid, cv=2, verbose=2,scoring='f1_macro') #cv = nr of folds
gs_mnb = gs_mnb.fit(x_train, y_train)


# =============================================================================
# # the results:
# =============================================================================
#for key in gs_mnb.best_estimator_.get_params().keys():
#    print(key,'_'*(15-len(key)),gs_mnb.best_estimator_.get_params()[key])
#
#best_paraSet = gs_mnb.best_estimator_.get_params()

import pandas as pd
cv_results = gs_mnb.cv_results_
results_df = pd.DataFrame({'rank': cv_results['rank_test_score'],
'params': cv_results['params'],
'cv score (mean)': cv_results['mean_test_score'],
'cv score (std)': cv_results['std_test_score']} )
aTokenizerIncluded_results_df = results_df.sort_values(by=['rank'], ascending=True)
pd.set_option('display.max_colwidth', 100)

# STore results In PKL with timestamp TBD

# TBD; params as  values, ranks as keys
df = aTokenizerIncluded_results_df.set_index('rank')
df.reset_index(inplace=True)
adic = df.to_dict('index')


