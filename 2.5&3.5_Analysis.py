# =============================================================================
# importing from 2.0 vectorization
# =============================================================================
#import pickle
#with open('output_2.0)Bow_Vectorizationψinput3.0_CrossValidation.pkl','rb') as f:  # Python 3: open(..., 'rb')
#   y,x_vec, x_vec_array,vectorizer = pickle.load(f)


import pandas as pd
from collections import Counter
#from spacy.lookups import Lookups
#lookups = Lookups()
#lookups.add_table("lemma_rules", {"noun": [["s", ""]]}) #all 3 instrumental ↓ 
from spacy.lemmatizer import Lemmatizer# to check why eg 'deceit' is not in the dict


##creating some clarity:
#X, Y,X_NONCONSONLY,X_CONSONLY = x,y,x_nonConsOnly, x_consOnly
#del x,y,x_nonConsOnly, x_consOnly # just so they dont appear in variable explorer

# =============================================================================
' START analysis) Input = "output_1.0_importPreprocessing"   Dataframe of TDM: '
# =============================================================================
import pandas as pd
##the whole internal corpus df:
#df4_x_vectorized = pd.DataFrame(x.toarray(), columns = vectorizer.get_feature_names())
#Cons only:
df4_x_vectorized = pd.DataFrame(x_vec_array, columns = vectorizer.get_feature_names())

df4_x_ConsOnly_vectorized = pd.DataFrame(x_consOnly_array, columns = vectorizer.get_feature_names())

df4_x_nonConsOnly_vectorized = pd.DataFrame(x_nonConsOnly_array, columns = vectorizer.get_feature_names())

# =============================================================================
' analyzing termPresence & Frequencies' #=============================================================================
# adding up the columns/terms along all documents(1axis/columns) gives the frequency of terms accross the whole specified corpus:
termFrequencies_x = df4_x_vectorized.sum().to_dict()

termFrequencies_Cons = df4_x_ConsOnly_vectorized.sum().to_dict()
# ↑ I doubleChecked (DC), it does show most common words in the corpus (without stopword removal 'the' is 1st): 

termFrequencies_nonCons = df4_x_nonConsOnly_vectorized.sum().to_dict()
#check: 0th column: term: '-PRON-' = 5 == 



#======================================================================== #
' length stats of documents: x, cons, nonCons                          '
#======================================================================== #
Len_nonCons = sum( termFrequencies_nonCons.values() )
Len_Cons = sum( termFrequencies_Cons.values() )
print( 'words entire corpus of label:\n nonCons:',Len_nonCons, '\nonCons:',Len_Cons, '\n nonCons have a factor more than cons of:' , Len_nonCons / Len_Cons, '\n\n, while there are 5 times as many docs with that label, is this due to longer average transcript of Cons?--checked: yes, 9.8k vs 3,4k cons nonCons respectively')

# lets look at the lengths of all the docs: CANOT BE DONE AFTER TRANSFORMATION IN A SPARSE MATRIX
Lengthlist_x = [len(nlp(document)) for document in X]
Lengthlist_nonConsOnly = [len(nlp(document)) for document in X_NONCONSONLY]
Lengthlist_x_consOnly = [len(nlp(document)) for document in X_CONSONLY]
from statistics import mean, stdev
print('\nmeans lengths in WORDS\n\n X:',mean(Lengthlist_x),'\nxnonCons:',mean(Lengthlist_nonConsOnly),'\nxcons:',mean(Lengthlist_x_consOnly) )



#======================================================================== #
' Term presence and frequency: x, cons, nonCons                           '
#======================================================================== #
# here I can check whether a word is present and how often it occurs:
TestTerm = 'respiratory'
try:
    print( termFrequencies_x[TestTerm] )
except KeyError:
    pass
#if something is not in the terms, try the stopwordlist:
TestTerm in Stopwords_endResult # eg


#here a list of terms can be checked for their frequencies in 2 corpora and compared
# I NEED TO NORMALIZE FOR DOC BEFORE I EXTRACT THEM TO THE TERM FREQUENCIES
def TermFrequency_checker(vocab_dict,vocab_dict_compare ):
    """ compares how often a word appears in 2 corpora"""
    print('\nchecking if these terms are in the vocab, and how often:\n')
    with open('conspiracyTerms_whatIthink.txt',encoding="utf-8") as f:
        ConspiracyTerms = f.readlines()
        
    for WORD in ConspiracyTerms: # I subjectively added these words to a txt file
        try:
            print( WORD.strip(),'-'*(15-len(WORD)),'   dict1:', int(vocab_dict[WORD.strip()]*1.6),'dict2:',vocab_dict_compare[WORD.strip()],'\n' )# add a NORMALIZE: number: ncons corpus have factor of 1.66 nr. words; DONT KNOW if the concept of that value is justified
        except KeyError:
                pass # doing nothing on exception
TermFrequency_checker(termFrequencies_Cons, termFrequencies_nonCons )


# =============================================================================
# #find out the doc(s) where a specified TERM occurs the most:
# =============================================================================
df4_x_vectorized['bill'].sort_values()
#write the doc to txt file to inspect:
with open("output_file.txt", "w", encoding="UTF-8") as output:
    output.write(str(x[295]))


# here I can see a list of most common ones:
countingWordOccurences_x = Counter(termFrequencies_x).most_common(100)
countingWordOccurences_Cons = Counter(termFrequencies_Cons).most_common(100)
countingWordOccurences_nonCons = Counter(termFrequencies_nonCons).most_common(100)



##for extracting most common bigrams (can also do this with only 2,2 paramter)
#bigramList = []
#for I in range(len(countingWordOccurences)):
#    if len( countingWordOccurences[I][0].split() ) >1:
#        listv.append(bigramList[I])
        
FeatureCountdf = pd.DataFrame(data=termFrequencies_Cons.values(),index=termFrequencies_Cons.keys(),columns=['one']) #instrumental only, for:
#WORDSTHATAPPEARXTIMES = FeatureCountdf[FeatureCountdf.one==2]

#======================================================================== #
' important for 1.	Determine ‘rare word frequency treshold’: n :         '
#======================================================================== #

# if the keys are reversed, it is basicially "countingWordOccurences_x" as eg, there is VALUE:1:term('know') that occurs KEY:10973 times
    #conversely: there are VALUE:453:terms('.., .. , 453') that occur KEY:3 times
term_counter = Counter(termFrequencies_x.values())
#eg there are VALUE:453 terms that occur 
sum ( term_counter.values() ) #check; this should add up to all features
#IF LIMITING THE MAX_FEATURE PARA, THE FEATURES/TERMS THAT WILL BE OMITTED WILL BE THE FEWEST OCCURING (1'S 2ND' ETC.)


#checking what column index/location/position a term is in the df:
df4_x_vectorized.columns.get_loc('bill')



##======================================================================== #
#' Analysis of the MNB classifier'
##======================================================================== #
#from sklearn.naive_bayes import MultinomialNB
##mnb_model=MultinomialNB() # ~==↓
#mnb_model= MultinomialNB(alpha= 1, fit_prior = True, class_prior= None) # apt the order of which class_prior paramters, are the order of: mnb_model.classes_
#
#
#print (f'\n mnb_model.class_prior is specified as: {mnb_model.class_prior}... ') # this works before fitting
#mnb_model.fit(x, y)
#import numpy as np
#print ('which results in a class prior of: ', np.exp(mnb_model.class_log_prior_  )) # this only works after fitting
#print('\n respecively for the classes:',mnb_model.classes_)
## use the EXP (invese log / euler) to convert the log back to probabilities:


'