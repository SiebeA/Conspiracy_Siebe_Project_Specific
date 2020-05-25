#======================================================================== #
' import the output of 1_preprocessing        '
#======================================================================== #
import numpy as np

#THESE ARE WITH THE LABELS 0-1
import pickle
with open('pickle\\output_1_importψpreprocessLABELS0-1;xyψdf3binarizedLenadjustedψxcons,noncons.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X, Y,DF3_BINARIZEDLABELSΨLENADJUSTED,X_CONSONLY,X_NONCONSONLY = pickle.load(f)
#x,y = X,Y

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
 tokenizer=my_cleaner_noLemma_noStop,
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


# make a customized dic of params I want to check:
vectorization_parasFiltered = vectorizer.get_params()
KEYS_TO_REMOVE = ['binary', 'decode_error' ,'dtype', 'encoding', 'input', 'strip_accents', 'vocabulary', 'analyzer', 'lowercase','norm','sublinear_tf']
for KEY in KEYS_TO_REMOVE:
    try: del vectorization_parasFiltered[KEY]
    except:         pass

print('\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---\n')
for KEY in vectorization_parasFiltered.keys():print(KEY,'_'*(15-len(KEY)),vectorization_parasFiltered[KEY])
'