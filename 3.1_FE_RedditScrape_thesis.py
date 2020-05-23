#======================================================================== #
'  Deprecated. Notes to be enhanced | Explored                         '
#======================================================================== #

'I can incorporate the vote balance, to see if certain pattersn (word counts/ ner counts, ner mentions) are associated with higher/lower vote balance'

#======================================================================== #
' setting up api connection                          '
#======================================================================== #
# https://WWW.reddit.com/r/Conspiracy/
import praw
reddit = praw.Reddit(client_id='xldY3pXZnBUlAA',
                     client_secret='llkoaWS3sV9KFQNS3J5CcYr74vg', password='WVwB6eiYfaVVn6C',
                     user_agent='prawtutorial', username='SiebeA')

#======================================================================== #
' comment import setup                         '
#======================================================================== #
subreddit = reddit.subreddit('conspiracy')


conversedict0 = {} # covnersedict = a dictionary; keys=ID values=LIST → [0]=votebalance originComment [1]=LIST→[0]=comment [1]DICT→[0]=votebalanceResponse [1] = lengthResponse [2] Responsecomment
Top_conspiracy = subreddit.top(time_filter='all',limit=3000) #the number of threads I want to include


import time
start_time = time.time()

#sa adding a list with dicts that contains metadata of the thread
listThreadInfo=[]
for submission in Top_conspiracy:
    threadInfo = {     "title":None ,     "voteBalance" :None ,     "subMissionID" :None }
    I=0
    
    if not submission.stickied:
        threadInfo.update(( ['title',submission.title], ['voteBalance', submission.ups], ['subMissionID',submission.id] ))
        listThreadInfo.append(threadInfo)
        I+=1
        
        #continuing the conversedict0:
        submission.comments.replace_more(limit=0) # https://youtu.be/KX2jvnQ3u60?t=702
        for comment in submission.comments.list():
            if comment.id not in conversedict0:
                conversedict0[comment.id] = [comment.ups,[comment.body,{}] ] 
               #ALTERNATIVE with tuples/labels :
#                if comment.parent() != submission.id:
#                    PARENT = str(comment.parent())
#                    conversedict0[PARENT][1][1][comment.id] = [("voteBalance:",comment.ups),("length:", len(comment.body.split()) ), comment.body]
                
                
                #Alternative without tuples (for DF)
                if comment.parent() != submission.id:
                    PARENT = str(comment.parent())
                    conversedict0[PARENT][1][1][comment.id] = [comment.ups,len(comment.body.split()) , comment.body]


print("--- %s seconds ---" % (time.time() - start_time)) #time it took

import pandas as pd# also storing in df: much more oversight
listThreadInfo_df = pd.DataFrame(listThreadInfo)

#======================================================================== #
'    subsetting criteria                       '
#======================================================================== #
import pandas as pd

origin_comments_criterium = []
for KEY in list(conversedict0.keys()):
    
        if conversedict0[KEY][0]>10 and len(conversedict0[KEY][1][0].split()) >100:#KEY[0] = votebalance ; KEY[1] = LEN_STRING
            origin_comments_criterium.append(( conversedict0[KEY][0], len( conversedict0[KEY][1][0].split()) , conversedict0[KEY][1][0] ))

#df for overview:
df_origin_criteriums = pd.DataFrame(origin_comments_criterium,columns= ['votebalance','length','post'])

#EDA
df_origin_criteriums.length.mean()

#---------------------------------------------------------------------
origin_comments_criterium_subset2 = []
for KEY in list(conversedict0.keys()):
    
        if conversedict0[KEY][0]>=6 and conversedict0[KEY][0]<=10 and len(conversedict0[KEY][1][0].split()) >100:
            origin_comments_criterium_subset2.append(( conversedict0[KEY][0], len( conversedict0[KEY][1][0].split()) , conversedict0[KEY][1][0] ))

df_origin_comments_criterium_subset2 = pd.DataFrame(origin_comments_criterium_subset2,columns= ['votebalance','length','post'])

#EDA
df_origin_comments_criterium_subset2.length.mean()

#---------------------------------------------------------------------

origin_comments_criterium_subset3 = []
for KEY in list(conversedict0.keys()):
    
        if conversedict0[KEY][0]>=1 and conversedict0[KEY][0]<=5 and len(conversedict0[KEY][1][0].split()) >=100:
            origin_comments_criterium_subset3.append(( conversedict0[KEY][0], len( conversedict0[KEY][1][0].split()) , conversedict0[KEY][1][0] ))

df_origin_comments_criterium_subset3 = pd.DataFrame(origin_comments_criterium_subset3,columns= ['votebalance','length','post'])

#EDA
df_origin_comments_criterium_subset3.length.mean()
df_origin_comments_criterium_subset3.length.count()


# =============================================================================
# analysis
# =============================================================================


len( df_origin_criteriums[df_origin_criteriums['length'].between(100, 200)] )

# =============================================================================
# Deprecated, seems like thse comments are duplicatives off origin comments
# =============================================================================
##making comment on comment list:
#Comments_onComments = []
#for KEY in conversedict0.keys(): 
#    for KEY2 in conversedict0[KEY][1][1]: # key 2 = 26 keys
#        Comments_onComments.append( conversedict0[KEY][1][1][KEY2] )
#        
#import pandas as pd
#df = pd.DataFrame(Comments_onComments,columns= ['votebalance','length','post'])
#
#
##and filtering on criteria:
#criteriaSatisfied_Comments_onComments = []
#for LIST in Comments_onComments:
#    if LIST[0][1] > 10 and LIST[1][1] > 100:
#        criteriaSatisfied_Comments_onComments.append(LIST)
# =============================================================================
# 
# =============================================================================





##======================================================================== #
#' External corpus (output file, input for NextStepPipeline                   '
##======================================================================== #
#reddit_corpus = origin_messages100PlusWords # entails: top 30 Conspiracy AllTime ⟶ originalComments ⟶ WordLEN >100 
#
##
### write to file
##with open("output_file.txt", "w") as output:
##    for I in a_external_corpus:
##        output.writelines("starttt\n"+str(I)+"\nenddd\n\n\n")
