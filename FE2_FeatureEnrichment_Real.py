
# =============================================================================
# Importing libraries (inc Wordvecmodel)
# =============================================================================
#nlp essentials:

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
#DIMENSIONS = '50'+'d'
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
' 5 machine learning without VECTOR ENRICHMENt'
# =============================================================================


# =============================================================================
'''enriching the Rare and out of vocab words:'''
# ============================================================================
import time
START_TIME = time.time()
###Enriching the Test-vector with nearest neighbors:
df = df4_x_vectorized # just for conenience
df_test = pd.DataFrame(x_test,columns=vectorizer.get_feature_names()) #test df
df_enriched = df_test.replace(df_test, 0) #creating the (here) empty enriched BOW

RARETERM_HYPERPARA= RARETERM_HYPERPARA
SIMILARITY_HYPERPARA = 0.80

#NEIGHBORS = []
NEIGHBORS_compare = [] # for comparing the neighbor len
try:
    for TERM in df_test:
        if np.count_nonzero(df[TERM]) <= RARETERM_HYPERPARA:#SET 'N' RARE-WORD HYPERPARAMATER          for TERM, which means that the term is unique in a doc, however, it can be used multiple times in that doc, as that is just the term that one uses to convey an idea
#            NEIGHBORS_compare.append(f'for this TERM: {TERM}') # here I can add the term (above neighbors)
            
#            print(f'\n "{TERM}" satisfies the "rare word paramater"')
            
            doc_occurence = np.where((df_test[TERM] >0))[0]
    #        print( 'and was used this many times in the WHOLE corpus:',int(np.sum(df[TERM])))
            
            for NEIGHBOR in model.similar_by_word(TERM, topn= 100): #SET 'K' NEIGHBORs HYPERPARAMATER
#!!!                print(f'{NEIGHBOR[0]}')
                            #[0] = TERM; [1]
                if NEIGHBOR[1]>SIMILARITY_HYPERPARA and NEIGHBOR[0] in df.columns: #PLUS MY EXTRA HYPER
#                if NEIGHBOR[0] in df.columns:
                    NEIGHBORS_compare.append(NEIGHBOR)
    #                print(f'\n TERM:{TERM} \n NEIGHBOR:{NEIGHBOR[0]} with a similarity of {NEIGHBOR[1]}')
                    #adding the total number of rare term occurences, in the Doc where the rare terms occurs, to the neighbor HYPERPARAMATER ?
                    for II in doc_occurence: #i.e. the doc in which the rare term occurs
                        df_enriched.at[II,NEIGHBOR[0]]+=(df_test[TERM].iloc[II]) # is SUM necessary here?;  also HYPERPARAMATER?
                        #BE CAREFULL WITH ADDING INT, AS IT rounds down
except KeyError:
    pass
print("--- %s minutes ---" % round((time.time() - START_TIME)/60,0))
print( len(NEIGHBORS), len(NEIGHBORS_compare) )

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

print(f' \n the difference of nonzero values in whole df as result of enrichment:\n {abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated))} \n compared to a total of 13600 cells in the df; that is {abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated))/13600*100} percent (NRC)')


# =============================================================================
# # MNB
# =============================================================================
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
nb_classifier = MultinomialNB()
nb_classifier.fit(x_train,y= y_train) # model.set_params(onehot__threshold=3.0) if want to one-hot encode only the terms that appear at least three times in the corpus; then the binarizer could be modelled as such
ŷ_nb = nb_classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ŷ_nb = nb_classifier.predict(pd.DataFrame(x_test))
for I in enumerate(ŷ_nb):  print(I)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
ŷ_probability = nb_classifier.predict_proba(x_test)

import numpy as np
#for i,j in zip(y_test, ŷ_nb):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )

print('\n*****', nb_classifier, '*****\n')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ_nb))
print(classification_report(y_test,ŷ_nb,zero_division=0))
print('accuracy_score',accuracy_score(y_test, ŷ_nb))

F1_scoreBaseline_mnb = round(f1_score(y_test,ŷ_nb,average='macro'),3)
print('\n***f1_score_macro :',F1_scoreBaseline_mnb,'***\n')

# =============================================================================
# MB with enrichment:
# =============================================================================
ŷ_enriched_mnb = nb_classifier.predict(df_aggregated) #  the predicted class
#for I in enumerate(ŷ_enriched_mnb):  print(I)
ŷ_enriched_probability = nb_classifier.predict_proba(df_aggregated)
#print(ŷ_enriched_probability)
print ('\n with the para:', nb_classifier)

ŷ_enriched_mnb = nb_classifier.predict(pd.DataFrame(x_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
#for i,j in zip(y_test, ŷ_enriched_mnb):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )
print('\n***', nb_classifier, '***\n')
#print(str( vectorizer)[:13])
print(confusion_matrix(y_test,ŷ_enriched_mnb))
print(classification_report(y_test,ŷ_enriched_mnb, zero_division=0))
#print('accuracy_score',accuracy_score(y_test, ŷ_enriched_mnb))
F1_score_enriched_mnb = round(f1_score(y_test,ŷ_enriched_mnb,average='macro'),3)
print('\n***f1_score_macro :',f1_score(y_test,ŷ_enriched_mnb,average='macro'),'***\n')


# =============================================================================
# # # Support Vector Machine
# =============================================================================
from sklearn import svm
svm_classifier = svm.SVC(C=1.0,kernel='linear', degree=3,gamma='auto')
svm_classifier.fit(x_train, y_train)

ŷ_svm = svm_classifier.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('\n\n',str( vectorizer)[:13])
print(classification_report(y_test,ŷ_svm, zero_division=0))
print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                            ['FN', 'TP']]) )
#print(str( vectorizer)[:13])
print('\n',confusion_matrix(y_test,ŷ_svm))
F1_scoreBaseline_svm = round(f1_score(y_test,ŷ_svm,average='macro'),3)
print('\n***f1_score_macro :',F1_scoreBaseline_svm,'***\n')


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
print('\n',confusion_matrix(y_test,ŷ_svm))
F1_score_enriched_SVM = round(f1_score(y_test,ŷ_svm_enriched,average='macro',),3)
print('\n***f1_score_macro :',F1_score_enriched_SVM,'***\n')


print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])


# =============================================================================
# analyzing differences in probabilities between enriched non enriched
# =============================================================================

ŷψY = pd.DataFrame(np.array((ŷ,y_test)).transpose(),columns=['ŷ','Y'])

EQUALITY = pd.DataFrame( ŷ_enriched_probability == ŷ_probability )
EQUAL_PRED = pd.DataFrame( ŷ_enriched == ŷ )

#putting it all together in a df: pull this up; 4 probabilities--2 for each of the tracks for each class; whether the 2 tracks were differnt, the actual , the predicted
compareProbs = pd.DataFrame( np.column_stack((ŷ_probability,
                                              ŷ_enriched_probability,
                                              EQUALITY[0], 
                                              y_test,
                                              ŷ,
                                              ŷ_enriched, 
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
print('\nnudges when pred and actual is diff:', Counter(compareProbs['effectEnrich_!=Y']))
#﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋

