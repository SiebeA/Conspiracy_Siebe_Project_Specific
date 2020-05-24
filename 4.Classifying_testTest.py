####======================================================================== #
###' following can be preloaded, before loop/gridsearch (uncomment 1st time then comment) '
####======================================================================== #
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
#
#import numpy as np
#
#import pickle
#
#import en_core_web_sm
#nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"]) 
##nlp.Defaults.stop_words |= {"soft","music",} #'|=' is to add several stopwords at once
##Stopwords_endResult = list(nlp.Defaults.stop_words)
###non lemma cleaner:
#def my_cleaner_noLemma(text):
#        return[token.lower_ for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_) <3 ) ] 
#        
#from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
#import pandas as pd 
#
#
#from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec#pretrained on wiki2014;
#import time
#
## =============================================================================
## #loading the word vectors models
## =============================================================================
##GLOVE
#DIMENSION = '200'+'d'
#GLOVE_FILE = datapath(f"C:\\Users\\Sa\\Google_Drive\\0_Education\\1_Masters\\WD_jupyter\\wordVectors\\glove.6B.{DIMENSION}.txt")
#WORD2VEC_GLOVE_FILE = get_tmpfile(f"glove.6B.{DIMENSION}.txt") # specify which d file is used here
#glove2word2vec(GLOVE_FILE,WORD2VEC_GLOVE_FILE)
##model:
#gloveModel = KeyedVectors.load_word2vec_format(WORD2VEC_GLOVE_FILE)


## Google news:
#from gensim.models import KeyedVectors
## Load vectors directly from the file
#Google_model = KeyedVectors.load_word2vec_format('wordvecmodels\\GoogleNews-vectors-negative300.bin', binary=True)


##BIG SELFTRAINED MODEL
#with open('pickle\\selfTrainedWord2vec4BIG.pkl','rb') as f:  # Python 3: open(..., 'rb')
#    selfTrainedw2vModel_big = pickle.load(f)
#print(len(selfTrainedw2vModel_big.wv.vocab))
#
#
#from sklearn.model_selection import train_test_split
#from collections import Counter # calculating the label distribution:
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
#from sklearn import svm        
##
###======================================================================== #
##' import the output of 1_preprocessing        '
###======================================================================== #
##THESE ARE WITH THE LABELS 0-1
#with open('pickle\\output_1_importψpreprocessLABELS0-1;xyψdf3binarizedLenadjustedψxcons,noncons.pkl','rb') as f:  # Python 3: open(..., 'rb')
#    Xǀtranscripts, Yǀlabels,DF3_BINARIZEDLABELSΨLENADJUSTED,X_CONSONLY,X_NONCONSONLY = pickle.load(f)
#print( len(Yǀlabels) , len(Xǀtranscripts) )


#======================================================================== #
''' TBD ↑ & TBE (to be experimented) experimenting:

- train the model only on the training data, thus vocabulary is only training terms; as this mirrors the real life situation?

- Try only to enrich unique neighbors
- including the enrichment product hyper 
- rare word need to be squared with max features
- when rare words are too few in comparison to max features --> loop to next max feature
- count the TP + TN for not having to do mental math
- 
'''


#======================================================================== #
'  !!! hyperpara definition    '
#======================================================================== #
WordvecModel = Google_model
#WordvecModel= gloveModel
print('WordvecModel_len',len(WordvecModel.wv.vocab))

# ↓ determines the ratio of rareterms/maxfeatures: i.e. 0.3 = 30% of the max features is the absolute rare term integer: eg 100 max features * 0.3 = rareterm = 30; this is because the ratio determines the enrichement %; now it is robust, if max feature changes, there is no problem
RatioHyper = 0.3

#INCLUDES MULTIPLICATION WITH THE SIMILARITY
SIMILARITY_HYPERPARA = 0#, 0.5, 0.6 , 0.7] # APT; the minium similarity of the neighbor, for it to be passed for check whether they are in vocabulary; LESS IMPORTANT SINCE THE ENRICHMENT EXPRESSION 
TOP_NEIGHBORS=50 # close cousin of similarity_hyperparamater, the top x neighbors to be passed for check whether they are in the vocabulary

# =============================================================================
# #hypers for classifying
# =============================================================================
from sklearn.naive_bayes import MultinomialNB#, BernoulliNB, GaussianNB
nb_classifier = MultinomialNB()

# =============================================================================
# hypers for vectorization
# =============================================================================
Vectorizer= TfidfVectorizer
MAXFEATURE =4000
MAX_DF = 0.9
MIN_DF = 1

vectorizer = Vectorizer(
 ngram_range=(1,1),
 tokenizer=my_cleaner_noLemma,
 max_features=MAXFEATURE, #sa note: this is important for my 'rare word frequency threshold'
 max_df = MAX_DF, #= 0.50 means "ignore terms that appear in more than 50% of the documents"; ie 1 means ignore no terms ; 0.0 ignore all terms
 min_df = MIN_DF,# 1 means "ignore terms that appear in less than 1 document: i.e. '1' would mean: not ignoring any terms"
 stop_words=None,#list(nlp.Defaults.stop_words), # max-df can take care of this???

 binary=False,#If True, all non zero counts are set to 1.
# use_idf= None,
 lowercase=True) #True by default: Convert all characters to lowercase before tokenizing.

#fit on the whole corpus (this means that the vocabulary is extracted from the whole corpus)
vectorizer.fit(Xǀtranscripts)

#Transform seperately: # DOES NOT SHOW UP IN VARIABLE EXPLORER
x_vec = vectorizer.transform(Xǀtranscripts) #whole internal corpus
x_vec_array = x_vec.toarray()
#print( 'this many docs and features:', x_vec_array.shape)

# =============================================================================
'splitting the vectorized dataset:'
# =============================================================================
SEED = 7 # used to be 7 # Note does not affact Test set, as there is nothing to split
x_train, x_validate, y_train, y_validate  = train_test_split(x_vec, Yǀlabels, test_size=0.2, random_state=SEED, shuffle=True, stratify=Yǀlabels)


# !!!=============================================================================
# modulating classifier and neighboring enrichment input (FIRST FIT VECTORIZER IN NEXT SECTION (put it here such that it is on top))
# =============================================================================
len( vectorizer.get_feature_names() ) # because wrong data shapes error keeps appearing
a_dataset = 'validation'

if a_dataset == 'validation':
    y = y_validate #  validation labels
    X = x_validate # validation transcripts
    df_X_ValǀTest = pd.DataFrame(x_validate.toarray(),columns=vectorizer.get_feature_names()) #Validatin df
    x_validate.shape
elif a_dataset == 'test':
    y = y_testtest # for testset labels
    X = x_vectesttest #testset trans
    df_X_ValǀTest = pd.DataFrame(x_vectesttest_array,columns=vectorizer.get_feature_names()) #test df




#len(vectorizer.get_feature_names())
#x_validate.shape
#x_vectesttest_array.shape


#======================================================================== #

# looping through the specified gridsearch hyperparas:

#if WordvecModel == selfTrainedw2vModel_small: # all for right name in file
#    WORDVEC_TYPE = 'selfsmall'
if WordvecModel == selfTrainedw2vModel_big:
    WORDVEC_TYPE = 'selfBIG'
elif WordvecModel == gloveModel:
    WORDVEC_TYPE = 'Glove'
elif WordvecModel == Google_model:
    WORDVEC_TYPE = 'google'

                
# =============================================================================
#         Checking the number of rare terms: important, as if too few, little will be enriched
# =============================================================================

df4_x_vectorized = pd.DataFrame(x_vec_array, columns = vectorizer.get_feature_names())
df4_x_vectorized.iloc[:,0].max() # CHECKS to see if the df is filled:

### some EDA on non-zero term occurences (dup:
a_nonZero_CountColumnsǀterms = pd.DataFrame(np.count_nonzero(df4_x_vectorized,axis=0),index=vectorizer.get_feature_names(),columns=['nonZeroCounts'])
a_nonZero_CountColumnsǀterms = a_nonZero_CountColumnsǀterms.sort_values('nonZeroCounts')
a_nonZero_CountColumnsǀterms = a_nonZero_CountColumnsǀterms.reset_index() #reset index low-high

#loop to determine what the RARETERM hyper needs to be for certain ratio of max feature:
for RARETERM in range(a_nonZero_CountColumnsǀterms.nonZeroCounts.max()):
    RATIO =len( a_nonZero_CountColumnsǀterms[a_nonZero_CountColumnsǀterms.nonZeroCounts<=RARETERM]) / len(a_nonZero_CountColumnsǀterms) # the ratio of I=rareword hyper
    if RATIO > RatioHyper:
        print('\n-to have the specified ratio rareword/max feature, the rareterm setting has been computed and set on:',RARETERM)
        break



# =============================================================================
'''enriching the Rare and out of vocab words:'''
# ============================================================================

df = df4_x_vectorized # just for convenience

#creating the enriched df; here with all zero values
ENRICHEDARRAY= np.array(df_X_ValǀTest)
ENRICHEDARRAY[ENRICHEDARRAY > 0] = 0 # IT WAS >255 ?????? what weird has to be 0
df_enriched = pd.DataFrame(ENRICHEDARRAY,columns=vectorizer.get_feature_names())

START_TIME1 = time.time()

NEIGHBORS = []
NEIGHBORS_NOTINBUILDING = []

for DOCINDEX in range(len( df_X_ValǀTest)): # loops through each of the docs
    try:
        print(DOCINDEX)
        START_TIME = time.time()
        for TERM in df_X_ValǀTest: #loops through each of the terms, of each of the doc
    #        print(TERM)
            if df_X_ValǀTest.iloc[DOCINDEX][TERM]>0 and np.count_nonzero(df[TERM]) <= RARETERM: #former checks if the term in the testvocaublary occurs in the specific test instance; latter checks how many non0/occurences the term has along all docs in training
    #                print(f'\n rareterm: "{TERM}" is in test doc')
                
# =============================================================================
                 for NEIGHBOR in WordvecModel.wv.similar_by_word(TERM, topn= TOP_NEIGHBORS): #SET 'K' NEIGHBORs HYPERPARAMATER
# =============================================================================
                    if NEIGHBOR[1]>SIMILARITY_HYPERPARA and NEIGHBOR[0] in df.columns:
                        NEIGHBORS.append(NEIGHBOR[0])
# =============================================================================
                        df_enriched.iloc[DOCINDEX][NEIGHBOR[0]] += NEIGHBOR[1] * df_X_ValǀTest.iloc[DOCINDEX][TERM]#↑adding a value that is the product of the similarity score(NEIGHBOR[1] * the value of the TERM in the respective DOC(df_X_ValǀTest.iloc[DOCINDEX][TERM]), to the NEIGHBOR of the rare term in the DOC where the rare term occurs
# HYPER PARAMATER                                        
# =============================================================================
                    else:
                        
                        NEIGHBORS_NOTINBUILDING.append(NEIGHBOR[0])
        print (" %s secs" % round((time.time() - START_TIME),0))
        print(f'len neighbors = {len(NEIGHBORS)}')
        print(f'len unique neighbors = {len(set(NEIGHBORS))}')
        

    except KeyError:
        print(KeyError, TERM)
        pass

print ("\n\n %s secs" % round((time.time() - START_TIME1),0))

print( len(NEIGHBORS) )

# =============================================================================
# #analyzing what happend in enrichment (and checking if things went right)
# =============================================================================
   # the enriched neighbors; from higehst to lowest TF IDF attribution
aa_highest_enriched_neighbors = df_enriched.max().sort_values(ascending=False).reset_index()
#printing the dfs:
print('\n test df \n',df_X_ValǀTest,'\n',)

print('\n df enriched df \n',df_enriched,'\n',)

df_aggregated = df_X_ValǀTest+df_enriched
print('\n aggregated df \n',df_aggregated,'\n',)

df_difference = df_aggregated-df_X_ValǀTest
print('df difference:\n',df_difference)

#the total nr of values change as a result of enrichment:
print(f'\n total summed values: \n df_X_ValǀTest.values: {df_X_ValǀTest.values.sum()} \n df_aggregated.values.sum: {df_aggregated.values.sum()}' )

Enriched_percentage = round(abs(np.count_nonzero(df_X_ValǀTest) - np.count_nonzero(df_aggregated)) /(df_X_ValǀTest.shape[0] * df_X_ValǀTest.shape[1])*100,1)
print(f' \n the difference of nonzero values in whole df as result of enrichment:\n {abs(np.count_nonzero(df_X_ValǀTest) - np.count_nonzero(df_aggregated))} \n compared to a total of {df_X_ValǀTest.shape[0] * df_X_ValǀTest.shape[1]} cells in the df \n that difference is  {Enriched_percentage} percent of total values in the BOW ')



# =============================================================================
# remember the vect paras:
# =============================================================================
        # make a customized dic of params I want to check:
vectorization_parasFiltered = vectorizer.get_params()
KEYS_TO_REMOVE = ['binary', 'decode_error' ,'dtype', 'encoding', 'input', 'strip_accents', 'vocabulary', 'analyzer', 'lowercase','norm','sublinear_tf']
for KEY in KEYS_TO_REMOVE:
    try: del vectorization_parasFiltered[KEY]
    except:         pass



#======================================================================== #
''' CLASSIFYING SECTION      '''
#======================================================================== #

 # a list of f1s and tp & tn 's for easy overview in the end:
f1s = []
TNψtp = []

#!!! =============================================================================
# # MNB BASELINE
# =============================================================================

nb_classifier = MultinomialNB(alpha=1, class_prior=None) #CLASS PRIOR
nb_classifier.fit(x_train,y_train) # model.set_params(onehot__threshold=3.0) if want to one-hot encode only the terms that appear at least three times in the corpus; then the binarizer could be modelled as such

ŷ_nb = nb_classifier.predict(X)

cR_nbBase = classification_report(y,ŷ_nb, zero_division=0)
print('\n',cR_nbBase)

#for i,j in zip(y, ŷ_nb):    print(i==j)
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )
print('\n*****', nb_classifier, '*****\n\nBASE NB')
#print(str( vectorizer)[:13])
print(confusion_matrix(y,ŷ_nb))

TNψfn_nb_base = confusion_matrix(y,ŷ_nb)[0,0] + confusion_matrix(y,ŷ_nb)[1,1]
print('tn+tp',TNψfn_nb_base)

F1_scoreBaseline_mnb = round(f1_score(y,ŷ_nb,average='macro'),3)
print('\n\n***f1_score_macro :',F1_scoreBaseline_mnb,'***\n\nBASE')
f1s.append(F1_scoreBaseline_mnb)


# =============================================================================
# NB with enrichment:
# =============================================================================
print ('\n with the para:', nb_classifier)

ŷ_nb_enriched = nb_classifier.predict(df_aggregated) # validation & test
        
cR_nbEnriched = classification_report(y,ŷ_nb_enriched, zero_division=0)
print(cR_nbEnriched)
   
print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                          ['FN', 'TP']]) )
print('\n***', nb_classifier, '***\n\nENRICHED NB')
#print(str( vectorizer)[:13])
print(confusion_matrix(y,ŷ_nb_enriched))
TNψfn_nb_enriched = confusion_matrix(y,ŷ_nb_enriched)[0,0] + confusion_matrix(y,ŷ_nb_enriched)[1,1]
print('tn+tp',TNψfn_nb_enriched)

F1_score_enriched_mnb = round(f1_score(y,ŷ_nb_enriched,average='macro'),3)
print('\n\n***f1_score_macro enriched NB:',f1_score(y,ŷ_nb_enriched,average='macro'),'***\n \n')
f1s.append(F1_score_enriched_mnb)

TNψtp.append((TNψfn_nb_base,TNψfn_nb_enriched))


# =============================================================================
# # # Support Vector Machine
# =============================================================================

svm_classifier = svm.SVC(C=2.8,kernel='linear', degree=3,gamma='auto',probability=True)
svm_classifier.fit(x_train, y_train)

ŷ_svm = svm_classifier.predict(X)

print('\n\n',str( vectorizer)[:13],'\n Base SVM:')

cR_svmBase = classification_report(y,ŷ_svm, zero_division=0)
print(cR_svmBase)

print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                            ['FN', 'TP']]) )
#print(str( vectorizer)[:13])
print('\n\n BASE SVM \n',confusion_matrix(y,ŷ_svm))
TNψfn_svm_base = confusion_matrix(y,ŷ_svm)[0,0] + confusion_matrix(y,ŷ_svm)[1,1]
print('tn+tp',TNψfn_svm_base)


F1_scoreBaseline_svm = round(f1_score(y,ŷ_svm,average='macro'),3)
print('\n\n***f1_score_macro Base SVM :',F1_scoreBaseline_svm,'***\n \n-------------')
f1s.append(F1_scoreBaseline_svm,  )


# =============================================================================
# SVM with ENRICHMENT
# =============================================================================
ŷ_svm_enriched = svm_classifier.predict(df_aggregated)
            
print('\n\n',str( vectorizer)[:13],'\n Enriched SVM:')

cR_svmEnriched = classification_report(y,ŷ_svm_enriched, zero_division=0)
print(cR_svmEnriched)


print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                            ['FN', 'TP']]) )
#print(str( vectorizer)[:13])
print('\n\n ENRICHED SVM \n\n',confusion_matrix(y,ŷ_svm_enriched))
TNψfn_svm_enriched = confusion_matrix(y,ŷ_svm_enriched)[0,0] + confusion_matrix(y,ŷ_svm_enriched)[1,1]
print('tn+tp',TNψfn_svm_enriched)

F1_score_enriched_SVM = round(f1_score(y,ŷ_svm_enriched,average='macro',),3)
print('\n\n***f1_score_macro enriched SVM :',F1_score_enriched_SVM,'***\n')

#
#print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
#for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])

f1s.append(F1_score_enriched_SVM)
TNψtp.append((TNψfn_svm_base,TNψfn_svm_enriched))


#_______________________________________________________________________
print('\n\n f1s:',f1s)
print('\n\n TNψtp:',TNψtp)
print(Enriched_percentage, 'percent of BOW cells were enriched\n')
print('max features', MAXFEATURE,'\n')
#print('svm C: ', svm_classifier.get_params()["C"])

#_______________________________________________________________________


#!!!======================================================================== #
' saving results and paras, somehow, function dont work if i load them, therefore just putting them here:      '
#======================================================================== #
# function for storing the baseline results of the 2 clfs in pickle hereafter
def paraPickleSaverBASE(clf):
    '''for the baseline tracks; both Nb & svm'''
    import pickle
    #:2 can be edited:
    VECTORIZER_TYPE = str(type(vectorizer))[-17:-9]
    MAX_FEATURES = vectorizer.get_params()['max_features'] #extracting params feature extraction
    NGRAM_RANGE = vectorizer.get_params()['ngram_range']
    MAX_DF = vectorizer.get_params()['max_df']
    MIN_DF = vectorizer.get_params()['min_df']
    CLF_name = str(clf)[:3].upper()
    F1_score = None
    TnTp = None
    C= None
    Alpha = None
    if clf == nb_classifier:
        F1_score = F1_scoreBaseline_mnb
        TnTp = TNψfn_nb_base
        Alpha = nb_classifier.get_params()['alpha']
    elif clf == svm_classifier:
        F1_score = F1_scoreBaseline_svm
        TnTp = TNψfn_svm_base
        C = svm_classifier.get_params()["C"]
        DEGREE = svm_classifier.get_params()["degree"]
    print('\n f1 score:', F1_score, '\n tn tp:', TnTp)
        
    from datetime import datetime
    TIME = datetime.now().strftime("_%d-%h-%H;%M;%S")
    PATH = 'C:\\Users\Sa\\WD_thesisPython_workdrive\\Text_Classification_Pipeline\\'
    # here I could add in {} WHAT I want to SHOW UP IN FILENAME.. COEFS AND METRICS.. ?WRITE TO EXCEL FILE?
    with open(PATH+f'{TIME};_Base_____f1={F1_score};_TnTp={TnTp};_CLF={CLF_name};__{VECTORIZER_TYPE};_max_features={MAX_FEATURES};_ngram={NGRAM_RANGE};_max_df={MAX_DF};_min_df={MIN_DF};__similarity={SIMILARITY_HYPERPARA};_rareterm={RARETERM};_ratioRareMax={RatioHyper};_c={C};_Alpha={Alpha};_Seed={SEED}_.thesis', 'wb') as f:
        pickle.dump([F1_score ,allParamaters],f)
        
#paraPickleSaverBASE(clf)

# function for storing the ENRICHED results of the 2 clfs in pickle hereafter
def paraPickleSaverENRICHED(clf):
    '''for the enriched track both Nb & svm'''
    import pickle
    #:2 can be edited:
    VECTORIZER_TYPE = str(type(vectorizer))[-17:-9]
    MAX_FEATURES = vectorizer.get_params()['max_features'] #extracting params feature extraction
    NGRAM_RANGE = vectorizer.get_params()['ngram_range']
    MAX_DF = vectorizer.get_params()['max_df']
    MIN_DF = vectorizer.get_params()['min_df']
    CLF_name = str(clf)[:3].upper()
    F1_score = None
    TnTp = None
    C=None
    Alpha = None
    if clf == nb_classifier:
        F1_score = F1_score_enriched_mnb
        TnTp = TNψfn_nb_enriched
        Alpha = nb_classifier.get_params()['alpha']
    elif clf == svm_classifier:
        F1_score = F1_score_enriched_SVM
        TnTp = TNψfn_svm_enriched
        C = svm_classifier.get_params()["C"]
        DEGREE = svm_classifier.get_params()["degree"]
        
    print('\n f1 score:', F1_score, '\n tn tp:', TnTp, '\n _enriched%:',round(abs(np.count_nonzero(df_X_ValǀTest) - np.count_nonzero(df_aggregated)) /(df_X_ValǀTest.shape[0] * df_X_ValǀTest.shape[1])*100),2)
        
    from datetime import datetime
    TIME = datetime.now().strftime("_%d-%h-%H;%M;%S")
    PATH = 'C:\\Users\Sa\\WD_thesisPython_workdrive\\Text_Classification_Pipeline\\'
    # here I could add in {} WHAT I want to SHOW UP IN FILENAME.. COEFS AND METRICS.. ?WRITE TO EXCEL FILE?
    with open(PATH+f'{TIME};_Enriched_f1={F1_score};_TnTp={TnTp};_CLF={CLF_name};__{VECTORIZER_TYPE};_max_features={MAX_FEATURES};_ngram={NGRAM_RANGE};_max_df={MAX_DF};_min_df={MIN_DF};__similar={SIMILARITY_HYPERPARA};_rareterm={RARETERM};_ratioRareMax={RatioHyper};_c={C};_Alpha={Alpha};_enriched={Enriched_percentage};_topn={TOP_NEIGHBORS};_d={DIMENSION};_Seed={SEED};_{WORDVEC_TYPE}_.thesis', 'wb') as f:
        pickle.dump([F1_score ,allParamaters],f)

# =============================================================================
# # a loop such that both the NB & SVM are saved in both tracks:
# =============================================================================
CLFS = [nb_classifier, svm_classifier]
for CLF in CLFS:
    # I wanted this: (and omitting the above functions) but does not work
#    from my_FunctionsModule_ForThesis import paraPickleSaverBASE
#    from my_FunctionsModule_ForThesis import paraPickleSaverENRICHED
    allParamaters = {'.raretermHyper':RARETERM,  'vectparas': vectorizer.get_params(), 'neighborsLEN':len(NEIGHBORS), 'neighborsLen_set':len(set(NEIGHBORS)), 'wordvecDimensions':DIMENSION,'.SIMILARITY_HYPERPARA':SIMILARITY_HYPERPARA,'clfParams': CLF.get_params()
 }
    paraPickleSaverBASE(CLF)
    print(str(CLF)[:3])
    paraPickleSaverENRICHED(CLF)
    print(str(CLF)[:3])
    print('****************************************************')

#some stats checks:     
x_vectesttest_array.shape
df_aggregated.shape
len(y)

##======================================================================== #
#'         '
##======================================================================== #
#    # trying out altenrative classifiers here, in this format:
#
## vectorizatin paramaters recap
#for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
#
#perceptron
#print( df_X_ValǀTest.shape)
#from sklearn.linear_model import Perceptron
#PERCEPTRON = Perceptron()
#PERCEPTRON.fit(x_train, y_train)
#ŷ_perceptron =          PERCEPTRON.predict(x_vectesttest_array)
#ŷ_perceptron_enriched = PERCEPTRON.predict(df_aggregated)
#print('\n',classification_report(y,ŷ_perceptron, zero_division=0))
#print('\n',classification_report(y,ŷ_perceptron_enriched, zero_division=0))
#print('perceptron\n',confusion_matrix(y,ŷ_perceptron))
#print('\n',confusion_matrix(y,ŷ_perceptron_enriched))
#
##MNb    
##print( df_X_ValǀTest.shape)
##print( df_aggregated.shape )
##nb_classifier = GaussianNB()  #=BernoulliNB() 
#nb_classifier =MultinomialNB()
#nb_classifier.fit(x_train, y_train)
#ŷ_nb_classifier = nb_classifier.predict(df_X_ValǀTest)
#ŷ_nb_classifier_enriched = nb_classifier.predict(df_aggregated)
#print('\n',classification_report(y_validate,ŷ_nb_classifier, zero_division=0))
#print('\n',classification_report(y_validate,ŷ_nb_classifier_enriched, zero_division=0))
#print('NB\n',confusion_matrix(y_validate,ŷ_nb_classifier))
#print('\n',confusion_matrix(y_validate,ŷ_nb_classifier_enriched))
#
##SVM    
#print( df_X_ValǀTest.shape)
#print( df_aggregated.shape )
#svm_classifier.fit(x_train, y_train)
#ŷ_svm_classifier = svm_classifier.predict(df_X_ValǀTest)
#ŷ_svm_classifier_enriched = svm_classifier.predict(df_aggregated)
#print('\n',classification_report(y_validate,ŷ_svm_classifier, zero_division=0))
#print('\n',classification_report(y_validate,ŷ_svm_classifier_enriched, zero_division=0))
#print('\n',confusion_matrix(y_validate,ŷ_svm_classifier))
#print('\n',confusion_matrix(y_validate,ŷ_svm_classifier_enriched))


