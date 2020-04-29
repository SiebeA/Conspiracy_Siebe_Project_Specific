
#======================================================================== #
' DRAGGGG:             df3_inputVectorization  and proceed'
#======================================================================== #

import en_core_web_sm

nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"]) 

nlp.Defaults.stop_words |= {"like","thing",} #'|=' is to add several stopwords at once
Stopwords_endResult = list(nlp.Defaults.stop_words)


def my_cleaner4(text): #.is_alpha already excludes digits..
        return[token.lemma_ for token in nlp(text) if not (token.is_stop or token.lemma_ in Stopwords_endResult or token.is_alpha==False or len(token.lemma_) <3 ) ] # token.lemma_ in stopwords_endResulS; because eg: thingâœ“ thingsâœ— ; however, leave token.is.stop in, otherwise you end up with: 'PRONOUN ~9k'

#======================================================================== #
' BOW Vectorization                           '
#======================================================================== #
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = CountVectorizer(ngram_range=(1,1), lowercase=False, tokenizer=my_cleaner4, stop_words=None, max_features=None, binary=False) 

#fit on the whole corpus (this means that the vocabulary is extracted from the whole corpus)
vectorizer.fit(x)

#Transform seperately: # DOES NOT SHOW UP IN VARIABLE EXPLORER
x_v = vectorizer.transform(x) #whole internal corpus
x_array = x_v.toarray()
#======================================================================== #
' ↑ output termFrequency Vectorization → ready for 3 Cross validation      '
#======================================================================== #


x_nonConsOnly_v = vectorizer.transform(x_nonConsOnly)
x_nonConsOnly_array = x_nonConsOnly_v.toarray()

x_consOnly_v    = vectorizer.transform(x_consOnly)
x_consOnly_array = x_consOnly_v.toarray()
for I in dir(): print(I) # sparse matrices dont show up in VE, but are in dir..

#======================================================================== #
' ↑ output 2.0 → ready for 2.5 analysis                          '
#======================================================================== #


#======================================================================== #
' Extracting NERS  (for EDA, and I tried them as features: bad results '
#======================================================================== #

def NER_df(corpus_listOfStrings):
    """ extracts a lengthNormalized NER dataframe of a corpus"""
    import en_core_web_sm
    nlp = en_core_web_sm.load(disable=["tagger", "parser"]) # ner required
    import pandas as pd
    from collections import Counter
    entsPerDocList = []
    for I in range(len(corpus_listOfStrings)):
        doc = nlp( corpus_listOfStrings[I] )
        length = len(doc) #instrumental ↓ 
        lengthFactor =length/4946 #for normalizing. NOT ADAPTIVE YET.
        ents = [token.label_ for token in doc.ents]
        entCountList = Counter(ents)
        d = dict(entCountList) #lengthfactor operation does not work othewise
        d.update((x, y / lengthFactor) for x, y in d.items()) #normalize for length doc
        entsPerDocList.append(d)
    return pd.DataFrame( entsPerDocList )

ner_x_lenNormalized = NER_df(x).fillna(0)#deal with the Nans
ner_cons_lenNormalized = NER_df(x_consOnly).fillna(0)#deal with the Nans
ner_nCons_lenNormalized = NER_df(x_nonConsOnly).fillna(0)#deal with the Nans

#create 1 comparison dataframe:
import pandas as pd
listv = [ ner_cons_lenNormalized.mean(axis=0) , ner_nCons_lenNormalized.mean(axis=0) ]
ner_Means_compare = pd.concat(listv,axis=1)
ner_Means_compare.columns = ['cons','nCons']
ner_Means_compare['factor'] = ner_Means_compare.cons / ner_Means_compare.nCons


# =============================================================================
# Scaling the ner df
# =============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(ner_x_lenNormalized))

print(scaler.mean_)
ner_x_lenNormalized.mean()

ner_x_lenNormalized_StandardScaled=  scaler.transform(ner_x_lenNormalized)
import pandas as pd
ner_x_lenNormalized_StandardScaled_df = pd.DataFrame(ner_x_lenNormalized_StandardScaled, columns = ner_x_lenNormalized.columns)





'