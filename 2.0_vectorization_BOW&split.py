#======================================================================== #
' import the output of 1_preprocessing        '
#======================================================================== #

import pickle
with open('output_1_importψpreprocess;xyψdf3binarizedLenadjustedψxcons,noncons.pkl','rb') as f:  # Python 3: open(..., 'rb')
    x, y,df3_binarizedLabelsψlenAdjusted,x_consOnly,x_nonConsOnly = pickle.load(f)


print('some checks of former processes outputs; \n -checking some cleaning: term in corpus?: ..:' )
for I in ['[music]', '[Music]', '[soft music]']:print('term: ',I,'_'*(15-len(I)), I in str(x) )

#======================================================================== #
' importing NLP essentials' 'option: cleaning to preprocessing?'
#======================================================================== #

import en_core_web_sm

nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"]) 

nlp.Defaults.stop_words |= {"like","thing",} #'|=' is to add several stopwords at once
Stopwords_endResult = list(nlp.Defaults.stop_words)

#non lemma cleaner:
def my_cleaner_noLemma(text):
        return[token.lower_ for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_) <3 ) ] 

#lemmatizing cleaner:
#def my_cleaner_lemma(text):
#        return[token.lemma_ for token in nlp(text) if not (token.is_stop or token.lemma_ in Stopwords_endResult or token.is_alpha==False or len(token.lemma_) <3 ) ] #.is_alpha already excludes digits...
## token.lemma_ in stopwords_endResulS; because eg: thingâœ“ thingsâœ— ; however, leave token.is.stop in, otherwise you end up with: 'PRONOUN ~9k'

#======================================================================== #
' BOW Vectorization; for classifiation input                         '
#======================================================================== #
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
vectorizer = CountVectorizer(
 ngram_range=(1,1),
 tokenizer=my_cleaner_noLemma,
 max_features=None, #sa note: this is important for my 'rare word frequency threshold'
 max_df = 1.0, #= 0.50 means "ignore terms that appear in more than 50% of the documents".
 min_df = 1,# 1 means "ignore terms that appear in less than 1 document: i.e. not ignoring any terms"
 stop_words=None,
 binary=False,#If True, all non zero counts are set to 1.
# use_idf= None,
 lowercase=True) #True by default: Convert all characters to lowercase before tokenizing.

#======================================================================== #
' TBD: make before after vocabulary; to observe what terms were omitted   !!!        '
#======================================================================== #
#fit on the whole corpus (this means that the vocabulary is extracted from the whole corpus)
vectorizer.fit(x)

#Transform seperately: # DOES NOT SHOW UP IN VARIABLE EXPLORER
x_vec = vectorizer.transform(x) #whole internal corpus
x_vec_array = x_vec.toarray()
x_vec_array[:,0].max()
#belongs in 2.5analysis, but put it here for convenience
import pandas as pd 
df4_x_vectorized = pd.DataFrame(x_vec_array, columns = vectorizer.get_feature_names())
# CHECKS to see if they are filled:
df4_x_vectorized.iloc[:,0].max()
df4_x_vectorized.aaa.idxmax()
df4_x_vectorized.iloc[:,0].equals( df4_x_vectorized.aaa)

print('\n the vectorization paramaters:\n\n---',str(type(vectorizer))[-17:-9],'---') #CHECKS
for KEY in vectorizer.get_params().keys():print(KEY,'_'*(15-len(KEY)),vectorizer.get_params()[KEY])

# =============================================================================
# #saving output to disk; MIGHT NOT BE SOUND, AS THE VECTORIZATION WILL VARY DEPENDING ON PARA CHOICE
# =============================================================================
import pickle
with open('output_2.0)Bow_Vectorizationψinput3.0_CrossValidation.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([y,x_vec, x_vec_array], f)
    
    
#======================================================================== #
' ↓ for analyze purposes     '
#======================================================================== #

x_nonConsOnly_v = vectorizer.transform(x_nonConsOnly)
x_nonConsOnly_array = x_nonConsOnly_v.toarray()

x_consOnly_v    = vectorizer.transform(x_consOnly)
x_consOnly_array = x_consOnly_v.toarray()
for I in dir(): print(I) # sparse matrices dont show up in VE, but are in dir..
print('\n the vectorization paramaters:\n---',str(type(vectorizer))[-17:-9],'---\n')# Checks
for KEY in vectorizer.get_params().keys():print(KEY,'_'*(15-len(KEY)),vectorizer.get_params()[KEY])




##======================================================================== #
#' Extracting NERS  (for EDA, and I tried them as features: bad results '
##======================================================================== #
#
#def NER_df(corpus_listOfStrings):
#    """ extracts a lengthNormalized NER dataframe of a corpus"""
#    import en_core_web_sm
#    nlp = en_core_web_sm.load(disable=["tagger", "parser"]) # ner required
#    import pandas as pd
#    from collections import Counter
#    entsPerDocList = []
#    for I in range(len(corpus_listOfStrings)):
#        doc = nlp( corpus_listOfStrings[I] )
#        length = len(doc) #instrumental ↓ 
#        lengthFactor =length/4946 #for normalizing. NOT ADAPTIVE YET.
#        ents = [token.label_ for token in doc.ents]
#        entCountList = Counter(ents)
#        d = dict(entCountList) #lengthfactor operation does not work othewise
#        d.update((x, y / lengthFactor) for x, y in d.items()) #normalize for length doc
#        entsPerDocList.append(d)
#    return pd.DataFrame( entsPerDocList )
#
#ner_x_lenNormalized = NER_df(x).fillna(0)#deal with the Nans
#ner_cons_lenNormalized = NER_df(x_consOnly).fillna(0)#deal with the Nans
#ner_nCons_lenNormalized = NER_df(x_nonConsOnly).fillna(0)#deal with the Nans
#
##create 1 comparison dataframe:
#import pandas as pd
#listv = [ ner_cons_lenNormalized.mean(axis=0) , ner_nCons_lenNormalized.mean(axis=0) ]
#ner_Means_compare = pd.concat(listv,axis=1)
#ner_Means_compare.columns = ['cons','nCons']
#ner_Means_compare['factor'] = ner_Means_compare.cons / ner_Means_compare.nCons
#
#
## =============================================================================
## Scaling the ner df
## =============================================================================
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#print(scaler.fit(ner_x_lenNormalized))
#
#print(scaler.mean_)
#ner_x_lenNormalized.mean()
#
#ner_x_lenNormalized_StandardScaled=  scaler.transform(ner_x_lenNormalized)
#import pandas as pd
#ner_x_lenNormalized_StandardScaled_df = pd.DataFrame(ner_x_lenNormalized_StandardScaled, columns = ner_x_lenNormalized.columns)



