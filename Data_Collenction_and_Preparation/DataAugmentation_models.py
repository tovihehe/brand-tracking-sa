import pandas as pd
from transformers import pipeline

'''
Data augmentation technique involves adding noise or perturbations to the text, 
such as by adding spelling errors or random word replacements. 
'''

### Define the path of all the .CSV datasets
neutral_tweets = 'datasets/neutrals_data.csv'

### Load the .CSV files into a Pandas dataframe
neutral_tweets = pd.read_csv(neutral_tweets)

list_neutral_tweets = list(neutral_tweets.tweet)

### Translate tweets from English to Spanish and vice versa using Hugging Face's models
translator1 = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translator2 = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def new_neutral_ins(tweet):
    
	response_spa = translator1(tweet)
	response_spa = response_spa[0]['translation_text']

	response_eng = translator2(response_spa)
	response_eng = response_eng[0]['translation_text']

	return response_eng

val = 0

# create an empty DataFrame
new_df = pd.DataFrame(columns=['tweet', 'sentiment'])
new_neutral_instances = []

for tweet in list_neutral_tweets:
	val+=1
	print(val)
	new_ins = new_neutral_ins(tweet)
	new_row = {'tweet': new_ins, 'sentiment': 2}
	new_neutral_instances.append(new_row)

new_df = new_df.append(new_neutral_instances, ignore_index=True)

new_df = (pd.concat([neutral_tweets, new_df], ignore_index=True)).drop_duplicates().dropna()
new_df.to_csv('datasets/neutrals_data.csv', index=False)