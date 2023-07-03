CREATE TABLE TWEETS (

    id_tweet VARCHAR PRIMARY KEY, 
    tweet VARCHAR(255) NOT NULL, 
    username VARCHAR(255) NOT NULL,
    time_created DATE NOT NULL, 
    sentiment VARCHAR(255) NOT NULL,
    polarity REAL NOT NULL,
    confidence REAL NOT NULL,
    link VARCHAR(255) NOT NULL
    
)