Import libraries to build Word2Vec model


from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

#TBD:
'''

'''
#﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋﹋

#======================================================================== #
' loading        '
#======================================================================== #
#import, storing into a list
import csv
#loading the raw comments that were downloaded from Google big query:
with open("subreddit_BIG_=conspiracyAND-created_utcGreater-1422490091-AND-LENGTHbodyGreater500.csv",encoding='utf8') as CSVFILE:
    readCSV = csv.reader(CSVFILE)
    listv = []
    for ROW in readCSV:
        listv.append(ROW)
del readCSV

listv = [''.join(X) for X in listv] # from list of lists, to list of strings
#list of strings that satisfy conditions:
listOfLengthSatisfy = [I.lower() for I in listv if len(I.split()) > 20] # 200 = 1.68 milion ; 260 =1,22
del listv

# =============================================================================
# lower diverges more terms HOWEVER, bill(cinton) & (dollar)bill , will diverge?!
# =============================================================================

#removing dups:
listOfLengthSatisfy = list(set(listOfLengthSatisfy))
#aa = listOfLengthSatisfy[:10]

#======================================================================== #
' Cleaning        '
#======================================================================== #

# the cleaning secti integrated in the subsequent sections, however, I couldnt figure it out timely
sentences = []
# Go through each teXt in turn
for COMMENT in range(len(listOfLengthSatisfy)):
        for SENTENCE in listOfLengthSatisfy[COMMENT].strip().split('\n\n'):
            sentences.append(SENTENCE)
sentences = [x for x in sentences if x != "" ]
del listOfLengthSatisfy


#aa = sentences[100:200] # not what is cleaned on, thats file1sttrimmed ; this is for reference of what is before cleaning

#removing unwanted patterns:
import re
file_lst_trimmed = [re.sub(r'[(!"“”#$%&*+,-./:;<=>?@^_`()|~=\[\])]|gt|http\S+', '', file) for file in sentences]
#add: www

                               
                               
del sentences                               
#file_lst_trimmed # check next time, after additionall filters, if you still get: “Bill', 0.5799861550331116),  ('[Hillary', 0.5703248381614685)

##temporary saving to disk
#import pickle
#with open('tempFile1sttrimmed.pkl', 'wb') as f:
#                        pickle.dump(aa,f)

#import pickle
#with open('tempFile1sttrimmed.pkl','rb') as f:  # Python 3: open(..., 'rb')
#    test = pickle.load(f)

#and splitting the words in the 2_sentences:
#file_lst_trimmed = [SENTENCE.split() for SENTENCE in file_lst_trimmed]
# or alternatively, this, as it gives feedback of progress:
for I in range(len(file_lst_trimmed)):
    file_lst_trimmed[I] = file_lst_trimmed[I].split()
    print(I)




#excluding the splitted sentences with less than 5 tokens:
lenfiltered = []
for I,COMMENT in enumerate(file_lst_trimmed):
#    print(I,len(COMMENT))
    print(I)
    if len(COMMENT) >5:
        lenfiltered.append(COMMENT)
del file_lst_trimmed
#print("--- %s seconds ---" % (time.time() - START_TIME))
##remove the empty strings in the lists:
#for I in range(len(2_sentences)):
#    2_sentences[I] = list(filter(None, 2_sentences[I]))

# =============================================================================
# # Ouput; ie input for wordvec Trainings
# =============================================================================


#======================================================================== #
'WARNING, THE FOLLOWING IS VERY USEFULL, HOWEVER!! IT NEEDS TO BE BOTH , OR NOT APPLIED ON HERE AND ON THE VECTORIZATION OF THE TRANSCRIPTS         '
#======================================================================== #
#from gensim.models.phrases import Phraser, Phrases
## Phrase Detection
## Give some common terms that can be ignored in phrase detection
## For example, 'state_of_affairs' will be detected because 'of' is provided here: 
#common_terms = ["of", "with", "without", "and", "or", "the", "a"]
## Create the relevant phrases from the list of 2_sentences:
#phrases = Phrases(lenfiltered, common_terms=common_terms)
## The Phraser object is used from now on to transform 2_sentences
#bigram = Phraser(phrases)
##print(bigram[lenfiltered[4]])
#
## Applying the Phraser to transform our 2_sentences is simply
#lenfiltered2 = list(bigram[lenfiltered])


print('ready to make it')
#======================================================================== #
' creating the word vec model         '
#======================================================================== #
from gensim.models import Word2Vec

import time
START_TIME = time.time()

modelW2V_selftrained = Word2Vec(lenfiltered, 
                 #sg is 0 by default, meaning CBow will be used
                 min_count=3,   # Ignore words that appear less than this
                 size=200,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus
print("--- %s seconds ---" % (time.time() - START_TIME))

# 1884 seconds for 826k 2_sentences | vocab of 154k
print(modelW2V_selftrained)

print(modelW2V_selftrained.vector_size)
print(len(modelW2V_selftrained.wv.vocab))





#saving to disk
import pickle
with open('selfTrainedWord2vec4BIG.pkl', 'wb') as f:
                        pickle.dump(modelW2V_selftrained,f)


#make this for comparing with other wordvec models:
aa = modelW2V_selftrained.similar_by_word('illuminati',topn=25)
import pandas as pd
aa = pd.DataFrame(aa,columns=['word','similarity_score'])
aa.to_clipboard() # easy to excel
