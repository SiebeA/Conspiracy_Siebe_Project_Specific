#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
##======================================================================== #
#' following can be preloaded, before loop/gridsearch (uncomment 1st time then comment) '
##======================================================================== #
#
#import numpy as np
#import pickle
#import en_core_web_sm
#nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"]) 
#nlp.Defaults.stop_words |= {"like","thing",} #'|=' is to add several stopwords at once
##Stopwords_endResult = list(nlp.Defaults.stop_words)
###non lemma cleaner:
#def my_cleaner_noLemma_noStop(text):
#        return[token.lower_ for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_) <3 ) ] 
        

#======================================================================== #
'  !!! hyperpara definition / gridsearch definition       '
#======================================================================== #

# NOTE: the RARETERM and MAXFEATURE hypers are very dependable, because they dictate how many neighbors will be sought and how much enrichment will be done...
# an example of what different paramater settings concerning RARETERM & MAXFEATURES bring about:

# e.g., when rareterm = 25: 
    # 2000 MAXFEATURE: 277  rare words = 14%
    # 4000 MAXFEATURE: 2018 rare words = 50%
    
        # this is because:# the higher the max feature the more values are sliced that occur the least times (so first terms that only appear 1 time are omitted; then 2 etc.)
            #i.e. the end of the index (high frequency occuring terms) of a 1,000 & 10,000 maxFeature index are the same; because these terms are the last to be omitted.
      
        
#hypers for enriching
#RARETERM= 3
SIMILARITY_HYPERPARA = 0.75

#hypers for classifying
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
nb_classifier = GaussianNB()

# =============================================================================
# hypers for vectorization
# =============================================================================
MAXFEATURES = [10000]#, 15000,20000,25000,30000,32324]# 4000, 5000, 6000]
RARETERMLIST = [12]#,7,9,12,15,18,24,26,29] #if % rare terms too low; go to next
DIMENSIONLIST = ['50'+'d','200'+'d']#,'200'+'d','100'+'d']

#======================================================================== #
''' TBD ↑ & TBE (to be experimented) experimenting:
- Try online to enrich unique neighbors
- including the enrichment product hyper 
- rare word need to be squared with max features
- when rare words are too few in comparison to max features --> loop to next max feature
- count the TP + TN for not having to do mental math
- 

'''
#======================================================================== #

# looping through the specified gridsearch hyperparas:
for MAXFEATURE in MAXFEATURES:
    for RARETERM in RARETERMLIST:
        print('\n\nthis many Rareterms:',RARETERM)
#    for DIMENSION in DIMENSIONLIST:   # comment this to check for rare word length by maxfeatureX
        #======================================================================== #
        ' import the output of 1_preprocessing        '
        #======================================================================== #
        #THESE ARE WITH THE LABELS 0-1
        with open('pickle\output_1_importψpreprocessLABELS0-1;xyψdf3binarizedLenadjustedψxcons,noncons.pkl','rb') as f:  # Python 3: open(..., 'rb')
            X, Y,DF3_BINARIZEDLABELSΨLENADJUSTED,X_CONSONLY,X_NONCONSONLY = pickle.load(f)
        
        
#        print('some checks of cleaning processes outputs; \n -checking some cleaning: term in corpus?: ..:' )
#        for I in ['[music]', '[Music]', '[soft music]']:print('term: ',I,'_'*(15-len(I)), I in str(X) )
        
        #======================================================================== #
        ' importing NLP essentials' 'option: cleaning to preprocessing?'
        #======================================================================== #
    
        ###lemmatizing cleaner:
        #Stopwords_endResult = []
        #def my_cleaner_lemma(text):
        #        return[token.lemma_ for token in nlp(text) if not (token.is_stop or token.lemma_ in Stopwords_endResult or token.is_alpha==False or len(token.lemma_) <3 ) ] #.is_alpha already excludes digits...
        
        
        #======================================================================== #
        ' BOW Vectorization; for classifiation input                         '
        #======================================================================== #
        from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
        vectorizer = TfidfVectorizer(
         ngram_range=(1,1),
         tokenizer=my_cleaner_noLemma_noStop,
         max_features=MAXFEATURE, #sa note: this is important for my 'rare word frequency threshold'
         max_df = 0.8, #= 0.50 means "ignore terms that appear in more than 50% of the documents".
         min_df = 1,# 1 means "ignore terms that appear in less than 1 document: i.e. '1' would mean: not ignoring any terms"
         stop_words=None,#list(nlp.Defaults.stop_words), # max-df can take care of this???
# =============================================================================
         binary=False,#If True, all non zero counts are set to 1.
# =============================================================================
        # use_idf= None,
         lowercase=True) #True by default: Convert all characters to lowercase before tokenizing.
        
            
        #fit on the whole corpus (this means that the vocabulary is extracted from the whole corpus)
        vectorizer.fit(X)
        
        
        #Transform seperately: # DOES NOT SHOW UP IN VARIABLE EXPLORER
        x_vec = vectorizer.transform(X) #whole internal corpus
        x_vec_array = x_vec.toarray()
        print( 'this many docs and features:', x_vec_array.shape)
        
        
        #ANALYZING , STOP WORDS GOT FLTERED HERE??::
#        print('\nand after tokenizing and checking:; \n -checking some cleaning: term in corpus?: ..:' )
#        for I in ['[music]', '[Music]', '[soft music]']:print('term: ',I,'_'*(15-len(I)), I in str(X) )
#        
        
#!!! =============================================================================
#         Checking the number of rare terms: important, as if too few, little will be enriched
# =============================================================================
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
        a_RareTErmsLen = len( a_nonZero_CountColumnsǀterms[a_nonZero_CountColumnsǀterms.nonZeroCounts<=RARETERM] )
        print('len rare words:',a_RareTErmsLen)
        print(MAXFEATURE)
        
        Ratio_rareVSMaxFeature = a_RareTErmsLen/MAXFEATURE
        print('rare words as the ratio of maxfeatures:',Ratio_rareVSMaxFeature)
#        if a_RareTErmsLen > 0.5*MAXFEATURE: # CAREFULL AS IT CAN STOP HERE IF NOT SATISFIED
#            print()
        
# =============================================================================
#         
# =============================================================================
        
        for DIMENSION in DIMENSIONLIST:
        
                #======================================================================== #
                ' wordvecs & enrich & clf        '
                #======================================================================== #
                
                #
                #=============================================================================
                # Importing libraries (inc Wordvecmodel)
                #=============================================================================
                
                #nlp essentials:
                
                #import spacy
                #import en_core_web_sm # the english spacy core (sm = small database)
                # for installing spacy & en_core; https://spacy.io/usage
                
                import time
                START_TIME = time.time()
                
                # wordvec (GloVe) model:
        #        import numpy as np
                #%matplotlib notebook
                import matplotlib.pyplot as plt
                plt.style.use('ggplot')
        #        from sklearn.manifold import TSNE
        #        from sklearn.decomposition import PCA
                from gensim.test.utils import datapath, get_tmpfile
                from gensim.models import KeyedVectors
                from gensim.scripts.glove2word2vec import glove2word2vec#pretrained on wiki2014; 400kTerms
                
                #loading the word vectors
                DIMENSION = DIMENSION
                GLOVE_FILE = datapath(f"C:\\Users\\Sa\\Google_Drive\\0_Education\\1_Masters\\WD_jupyter\\wordVectors\\glove.6B.{DIMENSION}.txt")
                WORD2VEC_GLOVE_FILE = get_tmpfile(f"glove.6B.{DIMENSION}.txt") # specify which d file is used here
                glove2word2vec(GLOVE_FILE,WORD2VEC_GLOVE_FILE)
                
                #model:
                model = KeyedVectors.load_word2vec_format(WORD2VEC_GLOVE_FILE)
                
                print("--- %s seconds ---" % (time.time() - START_TIME))
                
                
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

                import time
                START_TIME1 = time.time()
                
                NEIGHBORS = []
                
                for DOCINDEX in range(len( df_test)): # loops through each of the docs
                    try:
                        print(DOCINDEX)
                        START_TIME = time.time()
                        for TERM in df_test: #loops through each of the terms, of each of the doc
                    #        print(TERM)
                            if df_test.iloc[DOCINDEX][TERM]>0 and np.count_nonzero(df[TERM]) <= RARETERM: #checks how many non0/occurences the term has along all docs in training
                    #                print(f'\n rareterm: "{TERM}" is in test doc')
                                
#!!! =============================================================================
                                 for NEIGHBOR in model.similar_by_word(TERM, topn= 100): #SET 'K' NEIGHBORs HYPERPARAMATER
# =============================================================================
                                    if NEIGHBOR[1]>SIMILARITY_HYPERPARA and NEIGHBOR[0] in df.columns:
                                        NEIGHBORS.append(NEIGHBOR[0])
# =============================================================================
                                        df_enriched.iloc[DOCINDEX][NEIGHBOR[0]] += NEIGHBOR[1] * df_test.iloc[DOCINDEX][TERM]#↑adding a value that is the product of the similarity score(NEIGHBOR[1] * the value of the TERM in the respective DOC(df_test.iloc[DOCINDEX][TERM]), to the NEIGHBOR of the rare term in the DOC where the rare term occurs
# HYPER PARAMATER                                        
# =============================================================================
                        print (" %s secs" % round((time.time() - START_TIME),0))
                        print(f'len neighbors = {len(NEIGHBORS)}')
                        print(f'len unique neighbors = {len(set(NEIGHBORS))}')

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
                
                Enriched_percentage = round(abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated)) /(df_test.shape[0] * df_test.shape[1])*100,1)
                print(f' \n the difference of nonzero values in whole df as result of enrichment:\n {abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated))} \n compared to a total of {df_test.shape[0] * df_test.shape[1]} cells in the df \n that difference is  {Enriched_percentage} percent of total values in the BOW ')
                
                # =============================================================================
                # remember the vect paras:
                # =============================================================================
                        # make a customized dic of params I want to check:
                vectorization_parasFiltered = vectorizer.get_params()
                KEYS_TO_REMOVE = ['binary', 'decode_error' ,'dtype', 'encoding', 'input', 'strip_accents', 'vocabulary', 'analyzer', 'lowercase','norm','sublinear_tf']
                for KEY in KEYS_TO_REMOVE:
                    try: del vectorization_parasFiltered[KEY]
                    except:         pass
                
                print('\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
                for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
                
                print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n\n dimensions wordvectors: {DIMENSION}\n\n')
                for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
                
                # a list of f1s and tp & tn 's for easy overview in the end:
                f1s = []
                tnψtp = []
                
                #!!! =============================================================================
                # # MNB BASELINE
                # =============================================================================
        #        from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
                from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
        
                
                #nb_classifier = GaussianNB()
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
                
                
                tnψfn_nb_base = confusion_matrix(y_test,ŷ_nb)[0,0] + confusion_matrix(y_test,ŷ_nb)[1,1]
                print('tn+tp',tnψfn_nb_base)
                
                print(classification_report(y_test,ŷ_nb,zero_division=0))
                
                F1_scoreBaseline_mnb = round(f1_score(y_test,ŷ_nb,average='macro'),3)
                print('\n\n***f1_score_macro :',F1_scoreBaseline_mnb,'***\n\nBASE')
                f1s.append(F1_scoreBaseline_mnb)
                
                # =============================================================================
                # NB with enrichment:
                # =============================================================================
                ŷ_enriched_mb = nb_classifier.predict(df_aggregated) #  the predicted class
                #for I in enumerate(ŷ_enriched_mb):  print(I)
                ŷ_enriched_probability = nb_classifier.predict_proba(df_aggregated)
                #print(ŷ_enriched_probability)
                print ('\n with the para:', nb_classifier)
                
                ŷ_enriched_mb = nb_classifier.predict(pd.DataFrame(x_test))
                
                
                #for i,j in zip(y_test, ŷ_enriched_mb):    print(i==j)
                print('\n format of CM:\n', np.array([    ['TN', 'FP'],
                                                          ['FN', 'TP']]) )
                print('\n***', nb_classifier, '***\n\nENRICHED NB')
                #print(str( vectorizer)[:13])
                print(confusion_matrix(y_test,ŷ_enriched_mb))
                tnψfn_nb_enriched = confusion_matrix(y_test,ŷ_enriched_mb)[0,0] + confusion_matrix(y_test,ŷ_enriched_mb)[1,1]
                print('tn+tp',tnψfn_nb_enriched)
                
                print(classification_report(y_test,ŷ_enriched_mb, zero_division=0))
                #print('accuracy_score',accuracy_score(y_test, ŷ_enriched_mb))
                F1_score_enriched_mnb = round(f1_score(y_test,ŷ_enriched_mb,average='macro'),3)
                print('\n\n***f1_score_macro enriched NB:',f1_score(y_test,ŷ_enriched_mb,average='macro'),'***\n \n')
                f1s.append(F1_score_enriched_mnb)
                
                
                # =============================================================================
                # # # Support Vector Machine
                # =============================================================================
                from sklearn import svm
                svm_classifier = svm.SVC(C=1.0,kernel='linear', degree=3,gamma='auto',probability=True)
                svm_classifier.fit(x_train, y_train)
                
                ŷ_svm = svm_classifier.predict(x_test)
                
                
                print('\n\n',str( vectorizer)[:13])
                print(classification_report(y_test,ŷ_svm, zero_division=0))
                print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                                            ['FN', 'TP']]) )
                #print(str( vectorizer)[:13])
                print('\n\n BASE SVM \n\n',confusion_matrix(y_test,ŷ_svm))
                tnψfn_svm_base = confusion_matrix(y_test,ŷ_svm)[0,0] + confusion_matrix(y_test,ŷ_svm)[1,1]
                print('tn+tp',tnψfn_svm_base)
                
                
                F1_scoreBaseline_svm = round(f1_score(y_test,ŷ_svm,average='macro'),3)
                print('\n\n***f1_score_macro Base SVM :',F1_scoreBaseline_svm,'***\n \n-------------')
                f1s.append(F1_scoreBaseline_svm)
                # =============================================================================
                # SVM with enrichment
                # =============================================================================
                ŷ_svm_enriched = svm_classifier.predict(df_aggregated)
                
                
                print('\n\n',str( vectorizer)[:13])
                print(classification_report(y_test,ŷ_svm_enriched, zero_division=0))
                print('\n\n format of CM:\n', np.array([    ['TN', 'FP'],
                                                            ['FN', 'TP']]) )
                #print(str( vectorizer)[:13])
                print('\n\n ENRICHED SVM \n\n',confusion_matrix(y_test,ŷ_svm_enriched))
                tnψfn_svm_enriched = confusion_matrix(y_test,ŷ_svm_enriched)[0,0] + confusion_matrix(y_test,ŷ_svm_enriched)[1,1]
                print('tn+tp',tnψfn_svm_enriched)
                
                F1_score_enriched_SVM = round(f1_score(y_test,ŷ_svm_enriched,average='macro',),3)
                print('\n\n***f1_score_macro enriched SVM :',F1_score_enriched_SVM,'***\n')
                
                print('\n\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
                for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
                
                f1s.append(F1_score_enriched_SVM)
                tnψtp.append((tnψfn_nb_base,tnψfn_nb_enriched,tnψfn_svm_base,tnψfn_svm_enriched))
                
                #_______________________________________________________________________
                print('\n\n f1s:',f1s)
                print('\n\n tnψtp:',tnψtp)
                print(Enriched_percentage, 'percent of BOW cells were enriched')
                
                
                #_______________________________________________________________________
                
                
                #!!!======================================================================== #
                ' saving results and paras:        '
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
                    if clf == nb_classifier:
                        F1_score = F1_scoreBaseline_mnb
                        TnTp = tnψfn_nb_enriched
                    elif clf == svm_classifier:
                        F1_score = F1_scoreBaseline_svm
                        TnTp = tnψfn_svm_base
                    print('\n f1 score:', F1_score, '\n tn tp:', TnTp)
                        
                    from datetime import datetime
                    TIME = datetime.now().strftime("_%d-%h-%H;%M;%S")
                    PATH = 'C:\\Users\Sa\\WD_thesisPython_workdrive\\Text_Classification_Pipeline'
                    # here I could add in {} WHAT I want to SHOW UP IN FILENAME.. COEFS AND METRICS.. ?WRITE TO EXCEL FILE?
                    with open(PATH+f'\{TIME}_____Base_f1Macro={F1_score}_TnTp={TnTp}_Clf={CLF_name}__{VECTORIZER_TYPE}_max_features={MAX_FEATURES}_ngram_range={NGRAM_RANGE}_max_df={MAX_DF}_min_df={MIN_DF}__similarity={SIMILARITY_HYPERPARA}_rareterm={RARETERM}_ratioRareMax={Ratio_rareVSMaxFeature}', 'wb') as f:
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
                    if clf == nb_classifier:
                        F1_score = F1_score_enriched_mnb
                        TnTp = tnψfn_nb_enriched
                    elif clf == svm_classifier:
                        F1_score = F1_score_enriched_SVM
                        TnTp = tnψfn_svm_enriched
                    print('\n f1 score:', F1_score, '\n tn tp:', TnTp, '\n _enriched%:',round(abs(np.count_nonzero(df_test) - np.count_nonzero(df_aggregated)) /(df_test.shape[0] * df_test.shape[1])*100),2)
                        
                    from datetime import datetime
                    TIME = datetime.now().strftime("_%d-%h-%H;%M;%S")
                    PATH = 'C:\\Users\Sa\\WD_thesisPython_workdrive\\Text_Classification_Pipeline'
                    # here I could add in {} WHAT I want to SHOW UP IN FILENAME.. COEFS AND METRICS.. ?WRITE TO EXCEL FILE?
                    with open(PATH+f'\{TIME}_Enriched_f1Macro={F1_score}_TnTp={TnTp}_CLF={CLF_name}__{VECTORIZER_TYPE}_max_features={MAX_FEATURES}_ngram_range={NGRAM_RANGE}_max_df={MAX_DF}_min_df={MIN_DF}__similarity={SIMILARITY_HYPERPARA}_rareterm={RARETERM}_ratioRareMax={Ratio_rareVSMaxFeature}_enriched%={Enriched_percentage}', 'wb') as f:
                        pickle.dump([F1_score ,allParamaters],f)
                
                # =============================================================================
                # # a loop such that both the NB & SVM are saved in both tracks:
                # =============================================================================
                CLFS = [nb_classifier, svm_classifier]
                for CLF in CLFS:
                    # I wanted this: (and omitting the above functions) but does not work
                #    from my_FunctionsModule_ForThesis import paraPickleSaverBASE
                #    from my_FunctionsModule_ForThesis import paraPickleSaverENRICHED
                    allParamaters = {'.raretermHyper':RARETERM, 'raretermslen': a_RareTErmsLen, 'vectparas': vectorizer.get_params(), 'neighborsLEN':len(NEIGHBORS), 'neighborsLen_set':len(set(NEIGHBORS)), 'wordvecDimensions':DIMENSION,'.SIMILARITY_HYPERPARA':SIMILARITY_HYPERPARA,'clfParams': CLF.get_params()
                 }
                    paraPickleSaverBASE(CLF)
                    print(str(CLF)[:3])
                    paraPickleSaverENRICHED(CLF)
                    print(str(CLF)[:3])

'