##########################################################################
#                        Importing dependencies                          #
##########################################################################
import tweepy 
from credentials import consumer_key, consumer_secret, access_token, access_token_secret 
import datetime as dt
import re
import contractions
import emoji
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup 
import warnings
warnings.filterwarnings("ignore", category=UserWarning) ### Disable BeautifulSoup warnings
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import plotly.express as px
from manage_database import init_db, truncate_table, visualize_results
import sqlite3
from htbuilder import div, big, h2, styles 
from htbuilder.units import rem
import datetime
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from pandas.api.types import (
    is_object_dtype,
    is_numeric_dtype,
)

#########################################################################
#                    Validating the Credentials                         #
#########################################################################

### We use Oauth in tweepy for authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

####################################################################################
#       Function to preprocess the tweets for the sentiment analysis model         #
####################################################################################

def preprocessing(tweet):  
    
    ### Fix contractions
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
    tweet = re.sub(r'([\;\:\|•«_<>\n])', ' ', tweet)
    
    ### Remove trailing whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

####################################################################################
#                        Define the class SentimentClassifier                      #
#################################################################################### 

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        output = self.drop(pooled_output)
        return self.out(output)
    
    
####################################################################################
#    Define the model to load the weights model and get the sentiment results      #
#################################################################################### 

class Model:
    def __init__(self):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        classifier = SentimentClassifier(n_classes=3)
        
        ### Load the weights of the model
        classifier.load_state_dict(
            torch.load('best_model_state.bin', map_location=torch.device('cpu'))
        )

        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, tweet):
        
        ### Define the targets
        targets = ['positive', 'negative', 'neutral']
        
        ### Encode the tweet
        encoded_text = self.tokenizer.encode_plus(
                tweet,
                add_special_tokens     = True,
                max_length             = 128,
                return_token_type_ids  = False,
                return_attention_mask  = True,
                return_tensors         = 'pt',
                padding                = 'max_length',
                truncation             = True
        )

        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)
        
        ### Get the probabilities, confidence and predicted class
        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        return (
            targets[predicted_class],
            confidence.item(),
            dict(zip(targets, probabilities)),
        )

### Function to create an instance of the model
def get_model():
    model = Model()
    return model

### Get the link of the tweet
def get_tweet_url(tweet):
        return f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"
    
### Get the starting date of the tweets
def get_start_date(days_ago):
    return datetime.datetime.now() - datetime.timedelta(days=days_ago)

####################################################################################
#         Function to retrieve the tweets, while getting sentiment results         #
####################################################################################

def get_tweets_data(
    search_params
    , conn):
    
    start_date = str(get_start_date(search_params['days_ago']).date())

    query_list = [
        search_params['topic'],
        "-RT" if search_params['exclude_retweets'] else "",
        f"since:{start_date}",
        "-filter:replies" if search_params['exclude_replies'] else "",
        "-filter:nativeretweets" if search_params['exclude_retweets'] else "",
        f"min_replies:{search_params['min_replies']}",
        f"min_retweets:{search_params['min_retweets']}",
        f"min_faves:{search_params['min_faves']}",
    ]
    
    ### Join the query list to create the query string
    query_str = " ".join(query_list)
    
    ### Create an instance of the model
    model = get_model()
    
    ### Create a dataframe to store the tweets
    tweets = pd.DataFrame(
        columns = [
                    'tweet'
                   , 'user_followers_count'
                   , 'retweet_count'
                   , 'favorite_count'
                   , 'sentiment'
                   , 'confidence'
        ])
    
    ### Tweepy object Cursor to retrieve tweets that contain the input brand
    cursor = tweepy.Cursor(
        api.search_tweets
        , q=query_str
        , lang='en'
        , tweet_mode='extended'
    ).items(search_params['limit'])
    
    ### Loop through the tweets, get sentiment results and store them in the dataframe
    for t in cursor:
        print("Retrieving tweet...")
        print(t.full_text)
        
        sentiment, confidence, probabilites = model.predict(preprocessing(t.full_text))
        polarity = probabilites['positive'] - probabilites['negative'] / (probabilites['positive'] + probabilites['negative'] + probabilites['neutral'])
        id_tweet = t.id_str
        tweet = t.full_text
        username = t.user.screen_name
        time_created = t.created_at
        link = get_tweet_url(t)
        
        print("Sentiment: ", sentiment)
        print("Confidence: ", confidence)
        print("Polarity: ", polarity)
        
        print("------------------------------------------")
        print("Probabilites:")
        print("Positive: ", probabilites['positive'])
        print("Negative: ", probabilites['negative'])
        print("Neutral: ", probabilites['neutral'])
                
        ### Insert the data into the database  
        c = conn.cursor()
        c.execute("INSERT INTO TWEETS (id_tweet, tweet, username, time_created, sentiment, polarity, confidence, link) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                  (id_tweet, tweet, username, time_created, str(sentiment), polarity, confidence, link))
        conn.commit()
        
####################################################################################
#               Using several useful functions for displaying stuff                #
####################################################################################

COLOR_RED = "#FF4B4B"
COLOR_BLUE = "#1C83E1"
COLOR_CYAN = "#00C0F2"
COLOR_GREEN = "#00C781"

polarity_color = COLOR_BLUE
negative_color = COLOR_RED
positive_color = COLOR_GREEN
neutral_color = COLOR_CYAN

### Function to create a style
def display_dial(title, value, color):
    st.markdown(
        div(
            style=styles(
                text_align="center",
                color=color,
                padding=(rem(0.8), 0, rem(3), 0),
            )
        )(
            h2(style=styles(font_size=rem(0.8), font_weight=600, padding=0))(title),
            big(style=styles(font_size=rem(3), font_weight=800, line_height=1))(
                value
            ),
        ),
        unsafe_allow_html=True,
    )

####################################################################################
#                            Run the streamlit web app                             #
####################################################################################

def app():  

    ### Set page title
    st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer")
    
    ### Set page layout
    a, b = st.columns([1, 10])
    with a:
        st.text("")
        st.image("TwitterLogo.png", width=50)
        
    with b:
        st.title("Twitter Sentiment Analyzer")
            
    st.write("###### Please enter a brand to see the most recent Twitter sentiment related to it.")
    st.sidebar.markdown("Let's analyze tweets about brands!")
    
    relative_dates = {
        "1 day ago": 1,
        "1 week ago": 7,
        "2 weeks ago": 14,
        "1 month ago": 30,
    }
    
    ### Create a form to get the input from the user
    with st.form(key='form'):
        with st.sidebar:
            
            search_params = {}
            a, b = st.columns([1, 1])
            search_params["topic"] = a.text_input("Enter a brand to search for:")
            search_params["limit"] = b.slider("Tweet limit", 1, 1000, 100)

            a, b, c, d = st.columns([1, 1, 1, 1])
            search_params["min_replies"] = a.number_input("Minimum replies", 0, None, 0)
            search_params["min_retweets"] = b.number_input("Minimum retweets", 0, None, 0)
            search_params["min_faves"] = c.number_input("Minimum hearts", 0, None, 0)
            selected_rel_date = d.selectbox("Search from date", list(relative_dates.keys()), 3)
            search_params["days_ago"] = relative_dates[selected_rel_date]

            a, b = st.columns([1, 2])
            search_params["exclude_replies"] = a.checkbox("Exclude replies", False)
            search_params["exclude_retweets"] = b.checkbox("Exclude retweets", False)
            
            print(search_params)
            
            submit = st.form_submit_button('Search Tweets 🔎')
    
    if submit:
        ### Truncate the existing table
        truncate_table()
    
        ### SQLite database to store the results of the sentiment analysis and other metadata
        conn = sqlite3.connect('twitter_database.db')
        
        ### Get the tweets and store them in the database
        get_tweets_data(search_params, conn)
        
        ### Close the connection to the database
        conn.close()
    
    tweets = visualize_results()
    st.write("## Sentiment from the most recent ", len(tweets)," tweets")
    polarity, positives, neutrals, negatives = st.columns(4)
    
    with polarity:
        display_dial("POLARITY", f"{tweets['polarity'].mean():.2f}", polarity_color)
    
    with positives:
        display_dial("POSITIVES", f"{int(tweets['sentiment'].value_counts().get('positive', 0))}", positive_color)
    
    with neutrals:
        display_dial("NEUTRALS", f"{int(tweets['sentiment'].value_counts().get('neutral', 0))}", neutral_color)
    
    with negatives:
        display_dial("NEGATIVES", f"{int(tweets['sentiment'].value_counts().get('negative', 0))}", negative_color)
    
    if search_params["days_ago"] <= 1:
        timeUnit = "hours"
        
    elif search_params["days_ago"] <= 30:
        timeUnit = "monthdate"
        
    else:
        timeUnit = "yearmonthdate"
    
    polarity_plot, bar_chart = st.columns(2)
    
    ### BAR CHART
    with bar_chart:
        sentiment = tweets['sentiment'].value_counts()
        sentiment = pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
        fig = px.pie(sentiment, values='Tweets', names='Sentiment', color_discrete_sequence=[positive_color, neutral_color, negative_color])
        
        ### Set the size of the chart
        fig.update_traces(textposition='inside', textinfo='percent', insidetextfont=dict(color='white'))
        fig.update_layout(width=600, height=380)
        st.plotly_chart(fig, use_container_width=True)
    
    ### POLARITY PLOT
    with polarity_plot:
        chart = alt.Chart(tweets, title="Sentiment Polarity")
        
        ### Create a line chart to show the average polarity
        avg_polarity = chart.mark_line(interpolate="catmull-rom", tooltip=True,).encode(
            x=alt.X("time_created:T", timeUnit=timeUnit, title="Time created"),
            y=alt.Y("mean(polarity):Q", title="Polarity", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color(value=polarity_color),
        )
        
        ### Create a scatter plot to show the polarity values
        polarity_values = chart.mark_point(size=85, filled=True,).encode(
            x=alt.X("time_created:T", timeUnit=timeUnit, title="Time created"),
            y=alt.Y("polarity:Q", title="Polarity"),
            color=alt.Color(value=polarity_color + "99"),
            tooltip=alt.Tooltip(["tweet", "polarity", "sentiment"]),
            href="link",
        )

        st.altair_chart(avg_polarity + polarity_values, use_container_width=True)
    
    ### Display information about the results    
    with st.expander("ℹ️ How to interpret the results", expanded=False):
        st.write(
            """
            **Polarity**: 
            It is a value which lies in the range of [-1,1], where 1 means positive statement and -1 means a negative statement.
            """
    )
    st.write("")
        
    ### Import the stopwords
    stop_words = stopwords.words('english')
    
    ### Add more stop words
    stop_words.extend(['would', 'another', 'around', 'since', 'us', 'co', 'amp', 'rt', 't', 's', 're', 'm', 
                       've', 'll', 'd', 'ain', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 
                       'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])

    ### Create an instance of CountVectorizer
    vectorizer = CountVectorizer(stop_words=stop_words)

    ### Apply preprocessing to the tweets
    tweets['preprocessed_tweet'] = tweets['tweet'].apply(preprocessing)
    
    ### Get the word counts of the tweets
    word_counts = vectorizer.fit_transform(tweets['preprocessed_tweet'])

    ### Get the list of words
    words = vectorizer.get_feature_names_out()

    ### Calculate the total count for each word
    word_counts = word_counts.sum(axis=0)

    ### Create a DataFrame with the word counts
    word_counts_df = pd.DataFrame({'Word': words, 'Count': word_counts.tolist()[0]})
    
    ### Create a slider to specify the number of top words to display
    st.write("## Top terms in tweets")
    top = st.slider("Specify the number of top words to extract", 1, 20, 10)
    
    ### Get the top words
    top_words_counts_df = word_counts_df.sort_values(by='Count', ascending=False).head(top)
    
    ### COUNT PLOT FOR WORDS
    st.altair_chart(
    alt.Chart(top_words_counts_df)
    .mark_bar(tooltip=True)
    .encode(
        x="Count:Q",
        y=alt.Y("Word:N", sort="-x"),
        color=alt.Color(value=COLOR_BLUE),
    ),
    use_container_width=True,
    ) 
        

##############################################################################################
#                                      Main                                                  #
##############################################################################################
if __name__ == "__main__":
    app()
