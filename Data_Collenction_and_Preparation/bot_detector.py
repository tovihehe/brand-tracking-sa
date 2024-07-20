
### TESTEO DE BOT DETECTOR 
from credentials import API_TOKEN
import requests


### tweets examples
tweet1 = "on the other hand I have Netflix that I don't even use"  ### negative
tweet2 = "Rewatch She-ra on Netflix and nearly cry because I feel so bad for Catra. I feel to much with her pains."  ### neutral
tweet3 = "@archielizabeths cuz it wasn't supposed to be split at all netflix is just a cunt" ### negative
tweet4 = "@netflix You can fall asleep in Everything Everywhere All at Once, wake up 20 minutes later, and not miss a thing.  Wtf?" ### negative
tweet5 = "cunt https://t.co/v5ZDVb12D0"  ### negative
tweet6 = "@successfulroses i hate netflix"  ### negative

API_URL = "https://api-inference.huggingface.co/models/tdrenis/finetuned-bot-detector"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
list_tweets = [tweet1, tweet2, tweet3, tweet4, tweet5, tweet6]

### FALSE --- It's a human
### TRUE --- It's a bot

def bot_detector_api(tweet): 

    response = requests.request("POST", API_URL, headers=headers, data=tweet)
    dicts = response.json()[0]
    
    score_human = dicts[0]['score']
    score_bot = dicts[1]['score']
    
    print("Bot score:", score_bot)
    print("Human score:", score_human)
    
    if score_bot > 0.75:
        
        # print("It's a bot ")
        return True
        
    # print("HUMAN!!!")
    return False
    
for tweet in list_tweets: 
    print(bot_detector_api(tweet))