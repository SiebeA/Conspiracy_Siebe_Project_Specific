.798

#======================================================================== #
' import the output of 1_preprocessing        '
#======================================================================== #
import numpy as np

#THESE ARE WITH THE LABELS 0-1
import pickle
with open('pickle\output_1_importψpreprocessLABELS0-1;xyψdf3binarizedLenadjustedψxcons,noncons.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X, Y,DF3_BINARIZEDLABELSΨLENADJUSTED,X_CONSONLY,X_NONCONSONLY = pickle.load(f)


print('some checks of cleaning processes outputs; \n -checking some cleaning: term in corpus?: ..:' )
for I in ['[music]', '[Music]', '[soft music]']:print('term: ',I,'_'*(15-len(I)), I in str(X) )

#======================================================================== #
' importing NLP essentials' 'option: cleaning to preprocessing?'
#======================================================================== #

import en_core_web_sm

nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"]) 

nlp.Defaults.stop_words |= {"like","thing",} #'|=' is to add several stopwords at once
#Stopwords_endResult = list(nlp.Defaults.stop_words)

##non lemma cleaner:
def my_cleaner_noLemma_noStop(text):
        return[token.lower_ for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_) <3 ) ] 

###lemmatizing cleaner:
#Stopwords_endResult = []
#def my_cleaner_lemma(text):
#        return[token.lemma_ for token in nlp(text) if not (token.is_stop or token.lemma_ in Stopwords_endResult or token.is_alpha==False or len(token.lemma_) <3 ) ] #.is_alpha already excludes digits...


#======================================================================== #
' BOW Vectorization; for classifiation input                         '
#!!!======================================================================== #
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
vectorizer = TfidfVectorizer(
 ngram_range=(1,1),
 tokenizer=my_cleaner_lemma,
 max_features=20000, #sa note: this is important for my 'rare word frequency threshold'
 max_df = 0.8, #= 0.50 means "ignore terms that appear in more than 50% of the documents".
 min_df = 1,# 1 means "ignore terms that appear in less than 1 document: i.e. '1' would mean: not ignoring any terms"
 stop_words=None,#list(nlp.Defaults.stop_words), # max-df can take care of this???
 binary=False,#If True, all non zero counts are set to 1.
# use_idf= None,
 lowercase=True) #True by default: Convert all characters to lowercase before tokenizing.

    
#======================================================================== #
' TBD: make before after vocabulary; to observe what terms were omitted   !!!        '
#======================================================================== #
#fit on the whole corpus (this means that the vocabulary is extracted from the whole corpus)
vectorizer.fit(X)


#Transform seperately: # DOES NOT SHOW UP IN VARIABLE EXPLORER
x_vec = vectorizer.transform(X) #whole internal corpus
x_vec_array = x_vec.toarray()
print( '\nthis many docs and features:', x_vec_array.shape)


#ANALYZING , STOP WORDS GOT FLTERED HERE??::
print('\nand after tokenizing and checking:; \n -checking some cleaning: term in corpus?: ..:' )
for I in ['[music]', '[Music]', '[soft music]']:print('term: ',I,'_'*(15-len(I)), I in str(X) )

    
#belongs in 2.5analysis, but put it here for convenience
import pandas as pd 
df4_x_vectorized = pd.DataFrame(x_vec_array, columns = vectorizer.get_feature_names())
df4_x_vectorized.iloc[:,0].max() # CHECKS to see if the df is filled:
#df4_x_vectorized.aaa.idxmax()
#df4_x_vectorized.iloc[:,0].equals( df4_x_vectorized.aaa)



### some EDA on non-zero term occurences (dup:
a_nonZero_CountColumnsǀterms = pd.DataFrame(np.count_nonzero(df4_x_vectorized,axis=0),index=vectorizer.get_feature_names(),columns=['nonZeroCounts'])
a_nonZero_CountColumnsǀterms = a_nonZero_CountColumnsǀterms.sort_values('nonZeroCounts')
a_nonZero_CountColumnsǀterms = a_nonZero_CountColumnsǀterms.reset_index() #reset index low-high

#Nonzero == term occurs in this doc: this is handy to determine rare term value
#THINK THIS IS RELATED TO THE MAX/MIN_DF PARAMATER

#The maximum frequency of term uses 
rareterm_hyperpara = 15
a_RareTErmsLen = len( a_nonZero_CountColumnsǀterms[a_nonZero_CountColumnsǀterms.nonZeroCounts<=rareterm_hyperpara] )
print('\nlen rare words',a_RareTErmsLen)
#THE SAME FOR EITHER TF & IDF (because its non zero)


# make a customized dic of params I want to check:
vectorization_parasFiltered = vectorizer.get_params()
KEYS_TO_REMOVE = ['binary', 'decode_error' ,'dtype', 'encoding', 'input', 'strip_accents', 'vocabulary', 'analyzer', 'lowercase','norm','sublinear_tf']
for KEY in KEYS_TO_REMOVE:
    try: del vectorization_parasFiltered[KEY]
    except:         pass

print('\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
    

    
    #======================================================================== #
' (for classifier/CV input) split train test & vectorize ; should i need to record these params too?'
#======================================================================== #
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x_vec_array, Y, test_size=0.2, random_state=7, shuffle=True, stratify=Y)

#checking dimension
print('dimensions of the train, test sets:', x_train.shape, x_test.shape )

from collections import Counter # calculating the label distribution:
COUNTwhole,COUNTtest  = Counter(Y) ,Counter(y_test)
print( '\nCounting labels:\n whole dataset:',COUNTwhole,'ratio label 2 in whole Y',COUNTwhole[1]/(COUNTwhole[0]+COUNTwhole[1]),'\n train:',Counter(y_train),'\n test:',Counter(y_test), '\nratio label 2 in y_test:', COUNTtest[1]/(COUNTtest[0]+COUNTtest[1]) )


# =============================================================================
'''enriching the Rare and out of vocab words:'''
# ============================================================================

###Enriching the Test-vector with nearest neighbors:
df = df4_x_vectorized # just for conenience
df_test = pd.DataFrame(x_test,columns=vectorizer.get_feature_names()) #test df

#creating the enriched df; here with all zero values
ENRICHEDARRAY= np.array(df_test)
ENRICHEDARRAY[ENRICHEDARRAY > 255] = 0
df_enriched = pd.DataFrame(ENRICHEDARRAY,columns=vectorizer.get_feature_names())


#!!! HYPERS::=============================================================================
rareterm_hyperpara= 15
#count how many that involves:
a_RareTErmsLen = len( a_nonZero_CountColumnsǀterms[a_nonZero_CountColumnsǀterms.nonZeroCounts<=rareterm_hyperpara] )
print('\nlen rare words',a_RareTErmsLen)


SIMILARITY_HYPERPARA = 0.3

# =============================================================================
import time
START_TIME1 = time.time()

NEIGHBORS = []

for DOCINDEX in range(len( df_test)): # loops through each of the docs
    try:
        print(DOCINDEX)
        START_TIME = time.time()
        for TERM in df_test: #loops through each of the terms, of each of the doc
    #        print(TERM)
            if df_test.iloc[DOCINDEX][TERM]>0 and np.count_nonzero(df[TERM]) <= rareterm_hyperpara: #checks how many non0/occurences the term has along all docs in training
    #                print(f'\n rareterm: "{TERM}" is in test doc')
                
                for NEIGHBOR in model.similar_by_word(TERM, topn= 100): #SET 'K' NEIGHBORs HYPERPARAMATER
                    if NEIGHBOR[1]>SIMILARITY_HYPERPARA and NEIGHBOR[0] in df.columns:
                        NEIGHBORS.append(NEIGHBOR)
    
    #                        print(f'TERM:{TERM} \n NEIGHBOR:{NEIGHBOR[0]} with a similarity of {NEIGHBOR[1]}')
                        df_enriched.iloc[DOCINDEX][NEIGHBOR[0]] += NEIGHBOR[1] * df_test.iloc[DOCINDEX][TERM]
        print (" %s secs" % round((time.time() - START_TIME),0))
        print(f'len neighbors = {len(NEIGHBORS)}')
    # =============================================================================
    #                         #↑adding a value that is the product of the similarity score * the value of the TERM in the respective DOC, to the NEIGHBOR of the rare term in the DOC where the rare term occurs: 
    # =============================================================================
    except KeyError:
        print(KeyError, TERM)
        pass

print ("\n\n %s secs" % round((time.time() - START_TIME1),0))

print( len(NEIGHBORS) )

# =============================================================================
# #analyzing what happend in enrichment
# =============================================================================

#printing the dfs:
print('\n test df \n',df_test,'\n',)

print('\n df enriched df \n',df_enriched,'\n',)

df_aggregated = df_test+df_enriched
print('\n aggregated df \n',df_aggregated,'\n',)

df_difference = df_aggregated-df_test
print('df difference:\n',df_difference)

#the total nr of values change as a result of enrichment:
print(f'\n total summed values: \n df_test.values: {df_test.values.sum()} \n df_aggregated.values.sum: {df_aggregated.values.sum()}' )

print(f' \n the difference of nonzero values in whole df as result of enrichment:\n {abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated))} \n compared to a total of {df_test.shape[0] * df_test.shape[1]} cells in the df \n that difference is  {round(abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated)) /(df_test.shape[0] * df_test.shape[1])*100),2} percent of total values in the BOW ')

# =============================================================================
# remember the vect paras:
# =============================================================================
print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
# =============================================================================
# # MNB BASELINE
# =============================================================================
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np

nb_classifier = GaussianNB()
nb_classifier.fit(x_train,y= y_train) # model.set_params(onehot__threshold=3.0) if want to one-hot encode only the terms that appear at least three times in the corpus; then the binarizer could be modelled as such
ŷ_nb = nb_classifier.predict(x_test)

ŷ_nb = nb_classifier.predict(pd.DataFrame(x_test))

ŷ_probability = nb_classifier.predict_proba(x_test)

#for i,j in zip(y_test, ŷ_nb):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )

print('\n*****', nb_classifier, '*****\n\nBASE NB')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ_nb))
print(classification_report(y_test,ŷ_nb,zero_division=0))
print('accuracy_score',accuracy_score(y_test, ŷ_nb))

F1_scoreBaseline_mnb = round(f1_score(y_test,ŷ_nb,average='macro'),3)
print('\n***f1_score_macro :',F1_scoreBaseline_mnb,'***\n\nBASE')

# =============================================================================
# NB with enrichment:
# =============================================================================
ŷ_enriched_mb = nb_classifier.predict(df_aggregated) #  the predicted class
#for I in enumerate(ŷ_enriched_mb):  print(I)
ŷ_enriched_probability = nb_classifier.predict_proba(df_aggregated)
#print(ŷ_enriched_probability)
print ('\n with the para:', nb_classifier)

ŷ_enriched_mb = nb_classifier.predict(pd.DataFrame(x_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
#for i,j in zip(y_test, ŷ_enriched_mb):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )
    
print('\n***', nb_classifier, '***\n\nENRICHED NB')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ_enriched_mb))
print(classification_report(y_test,ŷ_enriched_mb, zero_division=0))
#print('accuracy_score',accuracy_score(y_test, ŷ_enriched_mb))
F1_score_enriched_mnb = round(f1_score(y_test,ŷ_enriched_mb,average='macro'),3)
print('\n***f1_score_macro :',f1_score(y_test,ŷ_enriched_mb,average='macro'),'***\n \n')   


# =============================================================================
# # # Support Vector Machine
# =============================================================================
from sklearn import svm
svm_classifier = svm.SVC(C=1.0,kernel='linear', degree=3,gamma='auto',probability=True)
svm_classifier.fit(x_train, y_train)

ŷ_svm = svm_classifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('\n\n',str( vectorizer)[:13])
print(classification_report(y_test,ŷ_svm, zero_division=0))
print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                            ['FN', 'TP']]) )
#print(str( vectorizer)[:13])
print('\n\n BASE SVM \n',confusion_matrix(y_test,ŷ_svm))
F1_scoreBaseline_svm = round(f1_score(y_test,ŷ_svm,average='macro'),3)
print('\n***f1_score_macro :',F1_scoreBaseline_svm,'***\n \n-------------')

# =============================================================================
# SVM with enrichment
# =============================================================================
ŷ_svm_enriched = svm_classifier.predict(df_aggregated)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('\n\n',str( vectorizer)[:13])
print(classification_report(y_test,ŷ_svm_enriched, zero_division=0))
print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                            ['FN', 'TP']]) )
#print(str( vectorizer)[:13])
print('\n\n ENRICHED SVM \n',confusion_matrix(y_test,ŷ_svm_enriched))
F1_score_enriched_SVM = round(f1_score(y_test,ŷ_svm_enriched,average='macro',),3)
print('\n***f1_score_macro :',F1_score_enriched_SVM,'***\n')


print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
