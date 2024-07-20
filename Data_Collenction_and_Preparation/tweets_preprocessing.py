
############################################################################################
#      This file contains the preprocessing function for the BERT and RoBERTa models       #
############################################################################################

import pandas as pd
import numpy as np
import re
import contractions
import emoji
from bs4 import BeautifulSoup 
import warnings 
warnings.filterwarnings("ignore", category=UserWarning) ### Disable BeautifulSoup warnings

### Function to preprocess the tweets for the sentiment analysis model
def preprocessing(tweet):  
    
    tweet = re.sub("’", "'", tweet)
    tweet = contractions.fix(tweet)
    
    ### Remove URLs
    tweet = re.sub(r'http\S+', '', tweet) 
    tweet = re.sub(r'facebook\S+', '', tweet)
    tweet = re.sub(r'pic.twitter\S+', '', tweet)
    tweet = re.sub(r'youtube.com\S+', '', tweet)
    tweet = re.sub(r'twitch.tv\S+', '', tweet)
        
    ### Remove HTML tags
    tweet = BeautifulSoup(tweet).get_text()
    
    ### Remove @ mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    ### Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)

    ### Decode emojis
    tweet = emoji.demojize(tweet, language='en') 
    
    ### Decode
    tweet = tweet.encode('ascii', 'ignore').decode('utf-8')
    
    ### Remove repeating characters
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    
    ### Convert to lowercase
    tweet = tweet.lower()
    
    ### Replace character emojis with their sentiment
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'happy', tweet)
    tweet = re.sub(r'(:\s?d|:-d|x-?D|x-?d)', 'happy', tweet)
    tweet = re.sub(r'(<3|:\*)', 'happy', tweet)
    tweet = re.sub(r'(;-?\)|;-?d|\(-?;)', 'happy', tweet)
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'sad', tweet)
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', 'sad', tweet)
    tweet = re.sub(r'(:,\||:\'\|)', 'sad', tweet)
    
    ### Remove some special characters
    tweet = re.sub(r'([\;\:\|•«<>\n])', ' ', tweet)
    
    ### Remove trailing whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

### Read the data from the .CSV file into a pandas dataframe
final_df = pd.read_csv('LESS_DATA.csv', encoding = 'ISO-8859-1')

### Apply the cleaning function to the 'tweet' column
final_df["preprocessed_tweet"] = final_df["tweet"].apply(preprocessing)

### Remove the column 'tweet'
final_df = final_df.drop(['tweet'], axis=1)

### Remove rows with empty strings
final_df = final_df[final_df['preprocessed_tweet'] != '']

### Remove duplicates and null values
final_df = final_df.drop_duplicates().dropna()

### Save the dataframe to a .CSV file
final_df.to_csv('datasets/final_df_preprocessed.csv', index=False)