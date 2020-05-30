
#======================================================================== #
' processing the new stratified  set         '
#======================================================================== #

import pickle
with open('pickle//training_data_raf.pickle','rb') as f:  # Python 3: open(..., 'rb')
    train = pickle.load(f)
with open('pickle//test_data_raf.pickle','rb') as f:  # Python 3: open(..., 'rb')
    test = pickle.load(f)

x_train = [STRING[0] for STRING in train]
y_train = [STRING[1] for STRING in train]

from collections import Counter
#label distribution train:
Count_yTrain = Counter(y_train)
for I in Count_yTrain: print('label:',I,"=",Count_yTrain[I],"=",(Count_yTrain[I])/len(y_train))

#Count_corpus = Counter(df_corpus.labels)
#
#for I in Count_corpus: print('label:',I,"=",Count_corpus[I],"=",(Count_corpus[I])/len(df_corpus.labels))

x_test = [STRING[0] for STRING in test]
y_test = [STRING[1] for STRING in test]

Count_ytest = Counter(y_test)
for I in Count_ytest: print('label:',I,"=",Count_ytest[I],"=",(Count_ytest[I])/len(y_test))


# =============================================================================
# #for EDA
# =============================================================================
Lenlist = [len(STRING[0].split()) for STRING in train]
Lenlist_test = [len(STRING[0].split()) for STRING in test]


#creating dfs for EDA NOT FOR PROCESSING THE DATA FURTHER
import numpy as np
import pandas as pd
df_train = pd.DataFrame(np.array((x_train,y_train,Lenlist)).transpose(), columns = ['transcripts', 'labels', 'wordLengths'])
df_train = df_train.astype({"labels": int, "wordLengths":int})

df_test = pd.DataFrame(np.array((x_test,y_test,Lenlist_test)).transpose(), columns = ['transcripts', 'labels', 'wordLengths'])
df_test = df_test.astype({"labels": int, "wordLengths":int})

df_corpus_pre = df_train.append(df_test)
#stats:
print(df_corpus_pre.wordLengths.mean())
print(df_corpus_pre.wordLengths.median())
print(df_corpus_pre.wordLengths.std())
print(df_corpus_pre.wordLengths.max())
print(df_corpus_pre.wordLengths.min())


#======================================================================== #
' Removing < 100 length trans:        '
#======================================================================== #

# for the train set
training_list= []
counter = 0
for I,J in zip( x_train , y_train):
    training_list.append(((len(I.split()),I,J)))

#deleting the <100
for J,I in enumerate(training_list):
    if I[0] <100:
        training_list.pop(J)
        y_train.pop(J) # extra check
        counter +=1
        print('\nindex that is removed',J)
        print('\n\nentire example:',I)
print('\n\n this many deleted',counter)

#checking if everyihing went well:
labelListCheck = [I[2] for I in training_list]
labelListCheck == y_train


# for the test set
test_list= []
counter = 0
for I,J,K in zip( x_test, x_test , y_test):
    test_list.append(((len(I.split()),K,J)))

for J,I in enumerate(test_list):
    if I[0] <100:
        test_list.pop(J)
        y_test.pop(J)
        counter +=1
        print('\nindex that is removed',J)
        print('\n\nentire example:',I)
print('\n\n this many deleted',counter)

#checking if everyihing went well:
labelListCheck = [I[1] for I in test_list]
labelListCheck == y_test

#after stats ; LENS ARE STILL CORRECT
Lenlist_train_after = [S[0] for S in training_list]
Lenlist_test_after = [S[0] for S in test_list]



# =============================================================================
# #new sets: OUTPUTS
# =============================================================================
x_train_after= [I[1] for I in training_list]
y_train_after=[I[2] for I in training_list]

x_test_after= [I[2] for I in test_list]
y_test_after=[I[1] for I in test_list]


# =============================================================================
# #After LEN correction EDA
# =============================================================================
import pandas as pd
import numpy as np
df_train = pd.DataFrame(np.array((x_train_after,y_train,y_train_after,Lenlist_train_after)).transpose(), columns = ['transcripts', 'labels','labelscontrol', 'wordLengths'])
df_train = df_train.astype({"labels": int, 'labelscontrol':int, "wordLengths":int})

df_test = pd.DataFrame(np.array((x_test_after,y_test,y_test_after,Lenlist_test_after)).transpose(), columns = ['transcripts', 'labels','labelscontrol', 'wordLengths'])
df_test = df_test.astype({"labels": int, 'labelscontrol':int, "wordLengths":int})


df_corpus_after = df_train.append(df_test)
#CHECKIN if the labels are correct
df_corpus_after.labels.equals(df_corpus_after.labelscontrol)


#stats:
print(df_corpus_after.wordLengths.mean())
print(df_corpus_after.wordLengths.median())
print(df_corpus_after.wordLengths.std())
print(df_corpus_after.wordLengths.max())
print(df_corpus_after.wordLengths.min())



from collections import Counter
#label distribution train:
Count_yTrain = Counter(y_train_after)
for I in Count_yTrain: print('label:',I,"=",Count_yTrain[I],"=",(Count_yTrain[I])/len(y_train))

Count_ytest = Counter(y_test_after)
for I in Count_ytest: print('label:',I,"=",Count_ytest[I],"=",(Count_ytest[I])/len(y_test))



#======================================================================== #
' saving before bracket version to disk        '
#======================================================================== #
#to disk:
import pickle
with open('theNewTestSplits_withBrackets.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_train_after, y_train_after, x_test_after, y_test_after], f)


#apply cleaner:
cleaned_training = BracketCleaner( [I for I in x_train_after])
cleaned_test = BracketCleaner( [I for I in x_test_after])

#export the cleaned brackets:
import pickle
with open('theNewTestSplits_without_Brackets.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([cleaned_training, cleaned_test,y_train_after,y_test_after ], f)



# =============================================================================
'1.2 PRE CLEANING; finding all words in brackets eg[Music] APT(tokenizers dont deal with them well'
' NOT COMP ; eg [soft music] not dealt with '
# =============================================================================


def BracketCleaner(listOfstringsToBeCleaned):
    import re
    listOfTranscripts_cleaned = []
    re_matchlist_all_docs = []
    Regex = r'\[\w+\]|\[\w+\s\w+\]'#this pattern removes tokens sa '[music]' 
    II=0
    for String in listOfstringsToBeCleaned:
    #    print(String)
        
        MATCHLIST = re.findall(Regex, String) # this is for 1 doc
    #    print(MATCHLIST)
        
        # and removing the bracket words
        TOKENS = String.split()
        TOKENS  = [word for word in TOKENS if word not in MATCHLIST]
        CleanedString = ' '.join(TOKENS)
        listOfTranscripts_cleaned.append(CleanedString)
        re_matchlist_all_docs.append((II,MATCHLIST))
        II+=1
    print(re_matchlist_all_docs, '\n\n↑ these tokens in the enumerated transcripts have been removed')
    return listOfTranscripts_cleaned

cleaned = BracketCleaner( [I for I in x_train_after])



training1 = [I[0] for I in training]

#test if the cleaning it was effective

for DOC in df1_NoDuplicatives.transcripts:
    if '[Music]' in DOC or '[theme music]' in DOC:
        print('it is stil there')

#print('[Music]' in df1_NoDuplicatives.transcripts[0]) # i know it was in 0


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
#df2_lenAdjusted = df1_NoDuplicatives[df1_NoDuplicatives.wordLengths>1000]
#
#
#from collections import Counter
#print('labels distribution: ', Counter(df2_lenAdjusted.labels) )

import pickle
# cleaned_training, cleaned_test,y_train_after,y_test_after
with open('pickle//theNewTestSplits_without_Brackets.pkl','rb') as f:  # Python 3: open(..., 'rb')
    x_train, x_test, y_train, y_test = pickle.load(f)


# =============================================================================
# #changing labels 1 to labels 0:
# #changing labels 3 to labels 1:
# =============================================================================

def Binarizer_new(Y):
    from collections import Counter
    COUNT_before = (Counter(Y))
    for I,J in enumerate(Y):
        if J == 1:
            Y[I] = 0
        elif J==2:
            Y[I] = 1
        elif J==3:
            Y[I] = 1
        else:
            'WE GOT A PROBLEM'
    COUNT_after = (Counter(Y))
    print('before',COUNT_before, '\nafter:',COUNT_after)


Binarizer_new(y_train)
Binarizer_new(y_test)


import pickle
with open('pickle//theNewTestSplits_Binarized_without_Brackets.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_train, y_train,x_test,y_test], f)




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






