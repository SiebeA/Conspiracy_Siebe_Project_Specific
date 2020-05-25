#======================================================================== #
' 1.1 import ALFANO df1: dup df: transcripts, labels (and lengths words       '
#========================================================================


# import the json file containing file were duplicatives already are taken care of
import pandas as pd
df1_NoDuplicatives = pd.read_json('z_df1_NoDuplicatives.json')
#reset index so that the old-duplicates-indices are dealt with:
df1_NoDuplicatives = df1_NoDuplicatives.reset_index() #could drop it with: drop=True
#setting column names:
df1_NoDuplicatives.columns = ['ids', 'transcripts', 'labels', 'wordLengths']



# =============================================================================
# robustnss checking
# =============================================================================
#check for dups in whole df
df1_NoDuplicatives = df1_NoDuplicatives.drop_duplicates()



# pairwise similarity check for checking dups or similarities in transcripts specifically:
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = """this is a test,
a dog walks
I go to bed
a dog walks""".splitlines()

vect = TfidfVectorizer(min_df=1, stop_words=None)                                                                                                                                                                                                   
tfidf = vect.fit_transform(corpus)
vect.get_feature_names()
#similarity sparse matrix:
pairwise_similarity = tfidf * tfidf.T # I dont know exactly what happened in process, but output is good
# to array:
arr = pairwise_similarity.toarray()
#filling the 1's:
np.fill_diagonal(arr, np.nan)  # result: 1 & 3 are the same 

result_idx = np.nanargmax(arr,axis=1)


# =============================================================================
'1.2 finding all words in brackets eg[Music] APT(tokenizers dont deal with them well'
' NOT COMP ; eg [soft music] not dealt with '
# =============================================================================
re_matchlist_all_docs = []
import re
Regex = r'\[\w+\]|\[\w+\s\w+\]'#this pattern removes tokens sa '[music]' 
II=0
for String in df1_NoDuplicatives.transcripts:
#    print(String)
    
    MATCHLIST = re.findall(Regex, String) # this is for 1 doc
#    print(MATCHLIST)
    
    # and removing the bracket words
    TOKENS = String.split()
    TOKENS  = [word for word in TOKENS if word not in MATCHLIST]
    CleanedString = ' '.join(TOKENS)
    df1_NoDuplicatives.transcripts[II] = CleanedString
    re_matchlist_all_docs.append((II,MATCHLIST))
    II+=1
print(re_matchlist_all_docs, '\n\n↑ these tokens in the enumerated transcripts have been removed')

#test if the cleaning it was effective

for DOC in df1_NoDuplicatives.transcripts:
    if '[Music]' in DOC or '[theme music]' in DOC:
        print('it is stil there')

print('[Music]' in df1_NoDuplicatives.transcripts[0]) # i know it was in 0


## NOT NECESSAry, but this is to locate their position
#matches = re.find(Regex, String, re.MULTILINE)
#for matchNum, match in enumerate(matches, start=1):
#    
#    print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
#    
#    for groupNum in range(0, len(match.groups())):
#        groupNum = groupNum + 1
#        
#        print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))



# =============================================================================
'1.3 Filter wordLengths: only lengths words >1000:'
# =============================================================================
df2_lenAdjusted = df1_NoDuplicatives[df1_NoDuplicatives.wordLengths>1000]


from collections import Counter
print('labels distribution: ', Counter(df2_lenAdjusted.labels) )


# =============================================================================
# #changing labels 1 to labels 0:
# #changing labels 3 to labels 1:
# =============================================================================
print()
df3_binarizedLabelsψlenAdjusted = df2_lenAdjusted # instrumental for↓ 

df3_binarizedLabelsψlenAdjusted.loc[df3_binarizedLabelsψlenAdjusted['labels'] == 1,'labels'] = 0
df3_binarizedLabelsψlenAdjusted.loc[df3_binarizedLabelsψlenAdjusted['labels'] == 2,'labels'] = 1
df3_binarizedLabelsψlenAdjusted.loc[df3_binarizedLabelsψlenAdjusted['labels'] == 3,'labels'] = 1

#check if it went correclty:
print('labels distribution: ', Counter(df2_lenAdjusted.labels) )


df3_binarizedLabelsψlenAdjusted = df3_binarizedLabelsψlenAdjusted.reset_index(drop=True) #resetting index
#rc
print('labels distribution: ', Counter(df3_binarizedLabelsψlenAdjusted.labels) )




# =============================================================================
#  Clf preparation
'1.4 ↓ stored under "output_1.0_importPreprocessing.spydata" '
# =============================================================================
#corpus: transcripts and labels
x = list(df3_binarizedLabelsψlenAdjusted.transcripts)
y = list(df3_binarizedLabelsψlenAdjusted.labels)


print('[music], in x:','[music]' in str(x) ) # just to check 1.2
print('[soft music], in x:','[soft music]' in str(x) ) # just to check 1.2
print('know, in x:','know' in str(x) ) # just to check 1.2


#isolating Cons and nonCons:
df3_consOnly    = df3_binarizedLabelsψlenAdjusted[df3_binarizedLabelsψlenAdjusted.labels==1]
x_consOnly      = list(df3_consOnly.transcripts)
#y_consOnly      = list(df3_consOnly.labels) # necessary?
    
df3_nonConsOnly = df3_binarizedLabelsψlenAdjusted[df3_binarizedLabelsψlenAdjusted.labels==0]
x_nonConsOnly   = list(df3_nonConsOnly.transcripts)
#y_nonConsOnly   = list(df3_nonConsOnly.labels)


#======================================================================== #
' Saving to disk                          '
#======================================================================== #

import pickle
with open('pickle//output_1_importψpreprocessLABELS0-1;xyψdf3binarizedLenadjustedψxcons,noncons.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x,y,df3_binarizedLabelsψlenAdjusted,x_consOnly,x_nonConsOnly], f)






