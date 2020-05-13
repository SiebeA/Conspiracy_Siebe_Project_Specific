#THIS COULD BE MISLEADING BECAUSE THE OUTPUT OF 2.0VECTORIZATION OUGHT TO CHANGE REGULARLY
#import pickle
#with open('output_2.0)Bow_VectorizationÏˆinput3.0_CrossValidation.pkl','rb') as f:  # Python 3: open(..., 'rb')
#    y, x_vec, x_array = pickle.load(f)



#======================================================================== #
' (for classifier/CV input) split train test & vectorize ; should i need to record these params too?'
#======================================================================== #
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x_vec_array, y, test_size=0.2, random_state=3, shuffle=True, stratify=y)

#checking dimension
print('dimensions of the train, test sets:', x_train.shape, x_test.shape )

from collections import Counter # calculating the label distribution:
COUNTwhole,COUNTtest  = Counter(y) ,Counter(y_test)
print( '\nCounting labels:\n whole dataset:',COUNTwhole,'ratio label 2 in whole y',COUNTwhole[2]/(COUNTwhole[1]+COUNTwhole[2]),'\ntrai:',Counter(y_train),'\n test:',Counter(y_test), 'ratio label 2 in y_test:', COUNTtest[2]/(COUNTtest[1]+COUNTtest[2]) )


#======================================================================== #
' Classifying and evaluating on test set                    '
#======================================================================== #

# =============================================================================
# NB classifier   
# =============================================================================
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
nb_classifier = MultinomialNB()
nb_classifier.fit(x_train,y= y_train) # model.set_params(onehot__threshold=3.0) if want to one-hot encode only the terms that appear at least three times in the corpus; then the binarizer could be modelled as such
ŷ = nb_classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

print('\n the vectorization paramaters:\n---',str(type(vectorizer))[-17:-9],'---\n')
for key in vectorizer.get_params().keys():print(key,'_'*(15-len(key)),vectorizer.get_params()[key])


import numpy as np
#for i,j in zip(y_test, ŷ):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )

print('\n***', nb_classifier, '***\n')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ))
print(classification_report(y_test,ŷ))
print('accuracy_score',accuracy_score(y_test, ŷ))
print('\n***f1_score_macro :',f1_score(y_test,ŷ,average='macro'),'***\n')







# =============================================================================
# # SVM Support Vector Machine Classifier
# =============================================================================
from sklearn import svm
svm = svm.SVC(C=1.0,kernel='linear', degree=3,gamma='auto')
svm.fit(x_train,y= y_train)

ŷ_svm = svm.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('\n format of CM:\n', np.array([    ['TN','FP'],
                                          ['FN','TP']]) )
print( 'svm: \n', confusion_matrix(y_test,ŷ_svm))
print(classification_report(y_test,ŷ_svm))
print('accuracy_score',accuracy_score(y_test,ŷ_svm))
print('f1_score_macro :',f1_score(y_test,ŷ_svm,average='macro'))



#======================================================================== #
''' Cross Validation  (still train/test required, but no longer validation)   '''
#======================================================================== #
#cross validation
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn.naive_bayes import MultinomialNB
scoreMetrics_available = SCORERS #available score metrics

a_F1Macros = [] # keeping track of MEANS of #=cv-f1 scores, of specified alpha's
a_F1Macros.append( ((str(type(vectorizer))[-17:]), vectorizer.get_params() ))

#for every alpha--so 30 each , there are 12 CV scores of which the mean is calculated
rangex = [round( x * 0.1, 2) for x in range(1, 31)]
for I in rangex: #for 30 alphas:
    print(I) #print alpha
    # 30 times:

mnb = MultinomialNB(alpha=0.0001, fit_prior = True, class_prior= None)
crossValidation_scores = cross_validate(
        estimator= mnb, 
        X= x_train, 
        y= y_train, 
        cv =5, # nr of cv-splits
        scoring=['accuracy','recall_macro', 'precision_macro' , 'f1_macro'])
#printing the means of the K cross validation evaluation outcomes:
for keyScores,valuesValues in zip(crossValidation_scores.keys(), crossValidation_scores.values() ):
    print('\n', keyScores,'\n mean: ', round(valuesValues.mean() ,4) , '\n std: ' , round( valuesValues.std() ,4) )
#storing in list with params used
a_F1Macros.append ( ( round( crossValidation_scores['test_f1_macro'].mean(),4) , mnb.get_params() ) ) # is this the mean of 12 trials?

print('highscore:', max( [I[0] for I in a_F1Macros[1:] ] ),'\n with alpha:') # I would be WRONG; this is last alpha


# =============================================================================
# The pivotal end result: 'a_F1Macros' is a list of 31 elements where the 0th are paramaters used of vectorization, remaininging 30 elements contain the MEAN MACRO f1 SCORE per a variable ALPHA and fixed other MNB paramaters

# â†• == #sa (I try to formalize the above method)

# a_F1Macros = LIST containing 31 tuplesâ†’[0]Paramater used [1:]c containing [0]mean of 12 cv splits for [1] MNB_params
# =============================================================================



#======================================================================== #
' Saving the model + results to the disk ; tBD: write to excel                   '
#======================================================================== #
# =============================================================================
# # saving to disk with the specified parameter definition, and timestamp, so that they are not overwritten
# =============================================================================
import pickle
#:2 can be edited:
VECTORIZER_TYPE = str(type(vectorizer))[-17:-9]
MAX_FEATURES = vectorizer.get_params()['max_features'] #extracting params feature extraction
NGRAM_RANGE = vectorizer.get_params()['ngram_range']
highest_f1_ofAlphas = max( [I[0] for I in a_F1Macros[1:] ] ) #maximum score of tried out alphas
from datetime import datetime
time = datetime.now().strftime("__%d-%h-%H;%M;%S")
PATH = 'C:\Users\Sa\WD_thesisPython_workdrive\Text_Classification_Pipeline\macroF1_Scores_crossValidation'
# here I could add in {} WHAT I want to SHOW UP IN FILENAME.. COEFS AND METRICS.. ?WRITE TO EXCEL FILE?
with open(PATH+f'\F1scores;{time}__HighestCVmeanofAlpha={highest_f1_ofAlphas}__{VECTORIZER_TYPE}_max_features={MAX_FEATURES}_ngram_range={NGRAM_RANGE}', 'wb') as f:pickle.dump(a_F1Macros, f)


##loading a pickled file:
#import pickle # NEED TO EXECUTE 2.0: the cleaner
#with open(PATH+'\F1scores;;Highest=0.7475;max_features=None;ngram_range=(1, 1)____28-Apr-16;18;57', 'rb') as f:
#    modelÏˆresults = pickle.load(f)
    



# =============================================================================
# Making a unrelated to CV seperate NB .. for getting cf-matrices (compraing with raf)
# =============================================================================









#======================================================================== #
'                           '
#======================================================================== #
