
##########################################################################
#               Functions to manage the SQLite database                  #
##########################################################################

import sqlite3
import pandas as pd

def init_db():
    
    ### Connect to the database
    conn = sqlite3.connect("twitter_database.db")
    c = conn.cursor()
    
    ### Open the schema.sql file & execute the schema script
    with open('schema.sql') as f:
        c.executescript(f.read())
      
    ### Commit the changes  
    conn.commit()
    
    ### Close the connection
    conn.close()
        
    
def truncate_table():
    
    ### Connect to the database (create it if it doesn't exist)
    conn = sqlite3.connect('twitter_database.db')

    ### Create a cursor object
    c = conn.cursor()

    ### Check if the table exists
    c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='TWEETS';''')

    if c.fetchone():
        
        ### If the table exists, truncate it
        c.execute('''DELETE FROM TWEETS''')
        conn.commit()
        print("Table truncated.")
        
    else:
        print("Table doesn't exist.")
        init_db()

    ### Close the connection
    conn.close()
    
    
def visualize_results():
    ### Connect to the database
    conn = sqlite3.connect('twitter_database.db')

    ### Read the data from the database into a pandas dataframe
    df = pd.read_sql_query("SELECT * from TWEETS", conn)

    ### Print the dataframe
    # print(df)

    ### Close the connection
    conn.close()
    
    return df
    
    
####### TEST

### Truncate table
# truncate_table()

### Initialize database
# init_db()

### Visualize results
# visualize_results()

