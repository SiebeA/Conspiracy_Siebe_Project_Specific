

# =============================================================================
# What I dont need for this
# =============================================================================

del x_nonConsOnly_array, x_consOnly_array, x_vec_array, df3_binarizedLabelsψlenAdjusted # just so they dont appear in variable explorer



import numpy as np
#%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot')



from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

#loading the word vectors
GLOVE_FILE = datapath("C:\\Users\\Sa\\Google_Drive\\0_Education\\1_Masters\\WD_jupyter\\wordVectors\\glove.6B.100d.txt")
WORD2VEC_GLOVE_FILE = get_tmpfile("glove.6B.100d.txt")
glove2word2vec(GLOVE_FILE,WORD2VEC_GLOVE_FILE)

#model:
model = KeyedVectors.load_word2vec_format(WORD2VEC_GLOVE_FILE)

#some similarity word action:
import pandas as pd
print(pd.DataFrame( model.similar_by_word('afflicted', topn= 50)) )

# df[0].to_clipboard(index=False)a


#======================================================================== #
' Creating enriched BOW  

      '
#======================================================================== #

# =============================================================================
'Extracting the rare terms in the originalBOW'
# =============================================================================

# checkassumption: terms that only appear in 1 document with minimum & maximum of 1 time
# formal: term occurs in <=1 document; total count Term <=1
rare_Terms1 = []
for KEY in termFrequencies_x:
    if termFrequencies_x[KEY] <2: # 'HYPERPARAMATER: 'rare word frequency threshold' 'k'
        rare_Terms1.append(KEY)

for TERM in rare_Terms1[-100:]:
    print( df4_x_vectorized[TERM].sort_values(ascending=False))
print(len(rare_Terms1))


## heck assumption: terms that only appear in only 1 document with a minimum of 1 time, maximum of N
# formal: term occurs in <=1 document; total count N
rare_Terms12 = []
for TERM in df4_x_vectorized.columns:
#    print(TERM)
    if np.count_nonzero(df4_x_vectorized[TERM]) <2: # 'HYPERPARAMATER: 'rare word frequency threshold' 'k'
        rare_Terms12.append(TERM)
        
for TERM in rare_Terms12[-10:]:
    print( df4_x_vectorized[TERM].sort_values(ascending=False))
print( len(rare_Terms12) )


# difference between the two: Terms that occur in 1 doc max, however, multiple times
rareTerms_difference = list( set(rare_Terms12) - set(rare_Terms1) )
for TERM in rareTerms_difference[-100:]:
    print( df4_x_vectorized[TERM].sort_values(ascending=False))


#======================================================================== #
' Neighboring terms of rare_Terms1        '
#======================================================================== #
#Calling the neighbors for visualization comprehension:
for TERM in rare_Terms1[:100]:
    try:
        print('term in my corpus:                 ', TERM, '\n neighbors:')
        for TUPLE in model.similar_by_word(TERM, topn= 2): # HYPERPARAMTER: nr of neighbors 'k'
            print(TUPLE[0])
        print()
    except KeyError:
        pass
        print('***********KEYERROR\n\n')


# =============================================================================
# #  storing the Neighboring terms: FOR RARETERMS
# =============================================================================

neighboring_terms = []

for TERM in rare_Terms1:
    try:
        for TUPLE in model.similar_by_word(TERM, topn= 1): # HYPERPARAMTER: nr of neighbors 'k'
            neighboring_terms.append( (TUPLE[0]) )
    except KeyError:
        pass
len(neighboring_terms)
print('nr of rare terms without neighbor in wordvecmodel:',len(neighboring_terms)-len(rare_Terms1))


# =============================================================================
# Creating the enriched BOW  
# =============================================================================

# creating a new DF out of the old df4:
df4_x_vectorized_enriched = df4_x_vectorized
#: setting al values in the vocab to 0:
for col in df4_x_vectorized_enriched.columns:
    df4_x_vectorized_enriched[col].values[:] = 0
    
#checking which neighboring terms are/not present in the original BOW:
neighboring_term_notinoriginalBOW = []
neighboring_term_IN_originalBOW = []
for I in neighboring_terms:
    if I not in df4_x_vectorized_enriched.columns:
        print(I)
        neighboring_term_notinoriginalBOW.append(I)
    elif I in df4_x_vectorized_enriched.columns:
        neighboring_term_IN_originalBOW.append(I)

#check some neighboring terms in the original BOW manually:
'slacker' in  df4_x_vectorized_enriched.columns
'nirvana' in  df4_x_vectorized_enriched.columns

#checking which doc that is: #MAKE SURE THAT DF IS NOT NULLED
df4_x_vectorized['nirvana'].sort_values(ascending=False)
'nirvana' in str(x) #extra check

for TERM in neighboring_term_IN_originalBOW:
    print( np.count_nonzero(df4_x_vectorized[TERM]) )


#======================================================================== #
'understanding how neighborin words helps classification:         '
#======================================================================== #
# with these up to k  elements assigned a term frequency of 1 and all other elements in the vector assigned the value 0. 

'cia' is occurs in doc 1      the doc is label 2
classifier models a function between 'cia' & label 2

'fbi' is neighbor


in the test set: docx contains term 'fbi'
'fbi' was not present in the training data, therefore it does not provide evidence for class 2
'cia' is a neighbor of 'fbi'
'cia' provides evidene for class 2


model.similar_by_word('wuhan', topn= 25)




#checking which loc a term is: 
#print( df4_x_vectorized_enriched.columns.get_loc('chronic'), df4_x_vectorized.columns.get_loc('chronic'))
#columnOfRareTerm = df4_x_vectorized['chronic']

termFrequencies_x['agree'] # = total frequency of WORDS
# frequency of TermOccurences i.e. nr of docs in which the term occurs at least once >=1
np.count_nonzero(df4_x_vectorized['agree']) # *To count the number of non-zeros of all rows np.count_nonzero(df, axis=0))


#for 2.5 analyss?
df4_x_vectorized['agree'].value_counts() # '269 examples with 0 counts, 43 with 1 count, etc)

# questions: 
    # What is rare: - in 1 doc OR - 1 time across all docs
    # for neighboring terms, +1 at al docs?
    #Should I only store 1?,
    # what if a neighboring word is not in the original BOW?


