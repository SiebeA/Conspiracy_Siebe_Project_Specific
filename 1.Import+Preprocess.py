# test
#======================================================================== #
' 1.1 import df1: dup df: transcripts, labels (and lengths words       '
#======================================================================== #





# import the json file containing file were duplicatives already are taken care of
import pandas as pd
df1_NoDuplicatives = pd.read_json('dict1_NoDuplicatives.json')
#reset index so that the old-duplicates-indices are dealt with:
df1_NoDuplicatives = df1_NoDuplicatives.reset_index() #could drop it with: drop=True
#setting column names:
df1_NoDuplicatives.columns = ['ids', 'transcripts', 'labels', 'wordLengths']

# =============================================================================
# robustnss checking
# =============================================================================
#check for dups:
df1_NoDuplicatives = df1_NoDuplicatives.drop_duplicates()


# =============================================================================
'1.2 finding all words in brackets eg[Music] APT(tokenizers dont deal with them well'
' NOT COMP ; eg [soft music] not dealt with '
# =============================================================================
Re_matchlist = []
import re
Regex = r"\[\w+\]"#this pattern removes tokens sa '[music]' 
II=0
for String in df1_NoDuplicatives.transcripts: 
    MATCHLIST = re.findall(Regex, String)
    
    # and removing the bracket words
    TOKENS = String.split()
    TOKENS  = [word for word in TOKENS if word not in MATCHLIST]
    CleanedString = ' '.join(TOKENS)
    df1_NoDuplicatives.transcripts[II] = CleanedString
    Re_matchlist.append((II,MATCHLIST))
    II+=1
print(Re_matchlist, '\n\n↑ these tokens in the enumerated transcripts have been removed')

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
# #changing labels 3 to labels 2:
# =============================================================================
print()
df3_binarizedLabelsψlenAdjusted = df2_lenAdjusted # instrumental for↓ 
df3_binarizedLabelsψlenAdjusted.loc[df3_binarizedLabelsψlenAdjusted['labels'] == 3,'labels'] = 2
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
print('[soft music]' in str(x) ) # just to check 1.2
#isolating Cons and nonCons:

df3_consOnly    = df3_binarizedLabelsψlenAdjusted[df3_binarizedLabelsψlenAdjusted.labels==2]
x_consOnly      = list(df3_consOnly.transcripts)
y_consOnly      = list(df3_consOnly.labels)
    
df3_nonConsOnly = df3_binarizedLabelsψlenAdjusted[df3_binarizedLabelsψlenAdjusted.labels==1]
x_nonConsOnly   = list(df3_nonConsOnly.transcripts)
y_nonConsOnly   = list(df3_nonConsOnly.labels)


#RC:
#list( df3_binarizedLAbelsψlenAdjusted.labels ) == 