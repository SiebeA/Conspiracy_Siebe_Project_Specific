#
## =============================================================================
## Importing libraries (inc Wordvecmodel)
## =============================================================================
##nlp essentials:



#import spacy
#import en_core_web_sm # the english spacy core (sm = small database)
## for installing spacy & en_core; https://spacy.io/usage
#
#import time
#START_TIME = time.time()
#
## wordvec (GloVe) model:
#import numpy as np
##%matplotlib notebook
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec#pretrained on wiki2014; 400kTerms
#
##loading the word vectors
#DIMENSIONS = '200'+'d'
#GLOVE_FILE = datapath(f"C:\\Users\\Sa\\Google_Drive\\0_Education\\1_Masters\\WD_jupyter\\wordVectors\\glove.6B.{DIMENSIONS}.txt")
#WORD2VEC_GLOVE_FILE = get_tmpfile(f"glove.6B.{DIMENSIONS}.txt") # specify which d file is used here
#glove2word2vec(GLOVE_FILE,WORD2VEC_GLOVE_FILE)
#
##model:
#model = KeyedVectors.load_word2vec_format(WORD2VEC_GLOVE_FILE)
#
#print("--- %s seconds ---" % (time.time() - START_TIME))


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
RARETERM_HYPERPARA= 3
#count how many that involves:
a_RareTErmsLen = len( a_nonZero_CountColumnsǀterms[a_nonZero_CountColumnsǀterms.nonZeroCounts<=RARETERM_HYPERPARA] )
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
            if df_test.iloc[DOCINDEX][TERM]>0 and np.count_nonzero(df[TERM]) <= RARETERM_HYPERPARA: #checks how many non0/occurences the term has along all docs in training
    #                print(f'\n rareterm: "{TERM}" is in test doc')
                
                for NEIGHBOR in model.similar_by_word(TERM, topn= 100): #SET 'K' NEIGHBORs HYPERPARAMATER
                    if NEIGHBOR[1]>SIMILARITY_HYPERPARA and NEIGHBOR[0] in df.columns:
                        NEIGHBORS.append(NEIGHBOR)
    
    #                        print(f'TERM:{TERM} \n NEIGHBOR:{NEIGHBOR[0]} with a similarity of {NEIGHBOR[1]}')
                        df_enriched.iloc[DOCINDEX][NEIGHBOR[0]] += NEIGHBOR[1] * df_test.iloc[DOCINDEX][TERM]
        print (" %s secs" % round((time.time() - START_TIME),0))
        print(f'len neighbors = {len(NEIGHBORS)}')
    # =============================================================================
    #        Another hyperpara (line95) (by me, not mentioned in heap_2017): #↑it adds a value that is the product of the similarity score * the value of the TERM in the respective DOC, to the NEIGHBOR of the rare term in the DOC where the rare term occurs: 
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np

nb_classifier = MultinomialNB()
nb_classifier.fit(x_train,y= y_train) # model.set_params(onehot__threshold=3.0) if want to one-hot encode only the terms that appear at least three times in the corpus; then the binarizer could be modelled as such
ŷ_nb = nb_classifier.predict(x_test)

ŷ_nb = nb_classifier.predict(pd.DataFrame(x_test))

ŷ_probability = nb_classifier.predict_proba(x_test)

#for i,j in zip(y_test, ŷ_nb):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )

print('\n*****', nb_classifier, '*****\nBASE')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ_nb))
print(classification_report(y_test,ŷ_nb,zero_division=0))
print('accuracy_score',accuracy_score(y_test, ŷ_nb))

F1_scoreBaseline_mnb = round(f1_score(y_test,ŷ_nb,average='macro'),3)
print('\n***f1_score_macro :',F1_scoreBaseline_mnb,'***\n')

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
print('\n***', nb_classifier, '***\nENRICHED')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ_enriched_mb))
print(classification_report(y_test,ŷ_enriched_mb, zero_division=0))
#print('accuracy_score',accuracy_score(y_test, ŷ_enriched_mb))
F1_score_enriched_mnb = round(f1_score(y_test,ŷ_enriched_mb,average='macro'),3)
print('\n***f1_score_macro :',f1_score(y_test,ŷ_enriched_mb,average='macro'),'***\n\n ENRICHED \n')   


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
print('\n\n BASE \n',confusion_matrix(y_test,ŷ_svm))
F1_scoreBaseline_svm = round(f1_score(y_test,ŷ_svm,average='macro'),3)
print('\n***f1_score_macro :',F1_scoreBaseline_svm,'***\n \n-------------')

# =============================================================================
# SVM with enrichment
# =============================================================================
ŷ_svm_enriched = svm_classifier.predict(df_aggregated)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('\n\n',str( vectorizer)[:13])
print(classification_report(y_test,ŷ_svm, zero_division=0))
print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                            ['FN', 'TP']]) )
#print(str( vectorizer)[:13])
print('\n\n ENRICHED \n',confusion_matrix(y_test,ŷ_svm_enriched))
F1_score_enriched_SVM = round(f1_score(y_test,ŷ_svm_enriched,average='macro',),3)
print('\n***f1_score_macro :',F1_score_enriched_SVM,'***\n')


print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])


# =============================================================================
# 
# =============================================================================
#analyzing svm probs:
svm_test_probs_base = svm_classifier.predict_proba(x_test)
svm_test_probs_enriched = svm_classifier.predict_proba(df_aggregated)
# =============================================================================
# analyzing differences in probabilities between enriched non enriched
# =============================================================================

ŷψY = pd.DataFrame(np.array((ŷ_nb ,y_test)).transpose(),columns=['ŷ','Y'])

EQUALITY = pd.DataFrame( ŷ_enriched_probability == ŷ_probability )
EQUAL_PRED = pd.DataFrame(y_test ==ŷ_nb)

#putting it all together in a df: pull this up; 4 probabilities--2 for each of the tracks for each class; whether the 2 tracks were differnt, the actual , the predicted
compareProbs = pd.DataFrame( np.column_stack((ŷ_probability,
                                              ŷ_enriched_probability,
                                              EQUALITY[0], 
                                              y_test,
                                              ŷ_nb,
                                              ŷ_enriched_mnb, 
                                              EQUAL_PRED)),columns=['nonEnr_prob:0', 'non1', 'enr_prob:0','enr1','equalityProbs','y_test','y_nonenr','y_enr','equalpred'])
compareProbs['equalityProbs']  = compareProbs['equalityProbs'].astype('bool') #convert to bool
compareProbs['equalpred']= compareProbs['equalpred'].astype('bool') 
compareProbs = compareProbs.drop(['non1','enr1'], 1)


# this function gives feedback whether the enriching had a good/bad result on the probability per doc:
NUDGE_LIST=[]
for Y,NON,ENR in zip(compareProbs.y_test,compareProbs['nonEnr_prob:0'],compareProbs['enr_prob:0']):
    if Y == 1 and ENR<NON or Y==0 and ENR>NON:  NUDGE_LIST.append('good_Nudge')
    else:   NUDGE_LIST.append('!!bad_nudge!!')

NUDGE_LIST2=[]
for Y,NON,ENR,NONŷ in zip(compareProbs.y_test,compareProbs['nonEnr_prob:0'],compareProbs['enr_prob:0'],compareProbs['y_enr'] ):
    if Y != NONŷ:
        if Y == 1 and ENR<NON or Y==0 and ENR>NON:  NUDGE_LIST2.append('good_Nudge')
        else:   NUDGE_LIST2.append('!!bad_nudge!!')
    else:
        NUDGE_LIST2.append('nvm')

#adding to the compare df:
compareProbs['effectEnrich'] = NUDGE_LIST
compareProbs['effectEnrich_when:ŷ!=Y'] = NUDGE_LIST2 #when prediction is other than actual
print('\n\n', Counter(compareProbs.effectEnrich))
print('\nnudges when pred and actual is diff:', Counter(compareProbs['effectEnrich_when:ŷ!=Y']))

#﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋

