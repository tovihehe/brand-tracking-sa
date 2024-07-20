print('-----------------------------------------------------------------------')
print("Hello World")
print('-----------------------------------------------------------------------')

print("GPU IS AVAILABLE", torch.cuda.is_available())

### Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import transformers
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments, EvalPrediction

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

### Read .csv file
df_all = pd.read_csv('datasets/final_df_preprocessed.csv', encoding='utf-8')
df_neutrals = pd.read_csv('datasets/final_df_neutrals_preprocessed.csv', encoding='utf-8')

df = pd.concat([df_all, df_neutrals])

### Transform column 'sentiment' to integers
df['sentiment'] = df['sentiment'].astype(int)
df = df.drop_duplicates().dropna()

##########################################################################################################

### Define number of samples per class
n_samples = 500000
n_samples_neutrals = 100000

### Group data by class and sample n_samples from each group
### df = df.groupby('sentiment').apply(lambda x: x.sample(n_samples)).reset_index(drop=True)

# group the data by the label column
groups = df.groupby('sentiment')

# define a function to select a specific number of rows from each group
def select_rows(group):
    sentiment = group.sentiment.unique()[0]
    if sentiment == 2:
        return group.sample(n=n_samples_neutrals, replace=True)
        
    elif sentiment in [1, 0]:
        return group.sample(n=n_samples, replace=True)

# apply the select_rows function to each group using the apply method & reset the index of the new dataframe
df = groups.apply(select_rows).reset_index(drop=True)

##########################################################################################################

### Count the values in the 'sentiment' column
value_counts = df['sentiment'].value_counts()
print("Value counts in column 'label':\n", value_counts)

### Drop rows with neutral sentiment
# df = df[df['sentiment'] != 2]


##########################################################################
#           Obtain the maximum length of the token sentences             #
##########################################################################

### Encode our concatenated data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_tweets = [tokenizer.encode(tweet, add_special_tokens=True) for tweet in df.preprocessed_tweet]

### Find the maximum length
lens = [len(tweet) for tweet in encoded_tweets]

print('--- Max length: ', max(lens))

#MAX_LEN = 128
MAX_LEN = max(lens)

##########################################################################
#                       Create TweetsDataset class                       #
##########################################################################

class TweetsDataset(Dataset):

  def __init__(self, tweets, targets, tokenizer, max_len):
    self.tweets    = tweets
    self.targets   = targets
    self.tokenizer = tokenizer
    self.max_len   = max_len
  
  def __len__(self):
    return len(self.tweets)
  
  def __getitem__(self, item):
    tweet = str(self.tweets[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      tweet,
      add_special_tokens     = True,
      max_length             = self.max_len,
      return_token_type_ids  = False,
      return_attention_mask  = True,
      return_tensors         = 'pt',
      padding                = 'max_length',
      truncation             = True
    )

    return {
      'tweet': tweet,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
    
##########################################################################
#                     Create Dataloaders function                        #
##########################################################################

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle):
  ds = TweetsDataset(
    tweets    = df.preprocessed_tweet.to_numpy(),
    targets   = df.sentiment.to_numpy(),
    tokenizer = tokenizer,
    max_len   = max_len
  )

  return DataLoader(
    ds,
    batch_size  = batch_size,
    num_workers = 4,
    shuffle = shuffle
  )

df_train, df_val = train_test_split(df, test_size = 0.1, random_state = RANDOM_SEED, shuffle = True)
print(df_train.shape, df_val.shape)
print("Created w/ success")

##########################################################################
#                            Define the model                            #
##########################################################################

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, dropout, freeze_bert = False):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.drop = nn.Dropout(p = dropout)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    ### Freeze the BERT model
    # if freeze_bert:
    #     for param in self.bert.parameters():
    #         param.requires_grad = False
  
  def forward(self, input_ids, attention_mask):
    pooled_output = self.bert(
      input_ids = input_ids,
      attention_mask = attention_mask
    )['pooler_output']

    output = self.drop(pooled_output)
    return self.out(output)
    
    
### Define the objective function to minimize
def objective(trial):
    
    ##########################################################################
    #                    Define hyperparameters to tune                      #
    ##########################################################################
    
    #learning_rate = trial.suggest_categorical("learning_rate", [5e-5, 3e-5, 2e-5])
    learning_rate = 2e-5
    
    #num_epochs = trial.suggest_categorical("num_epochs", [3])
    num_epochs = 3
    
    #batch_size = trial.suggest_categorical("batch_size", [32])
    batch_size = 16
    
    #eps = trial.suggest_categorical("eps", [1e-8, 1e-7, 1e-6])
    percentage = trial.suggest_categorical("percentage", [0, 0.05, 0.1])
    dropout_prob = trial.suggest_categorical("drop_out", [0.1, 0.3, 0.5])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #MAX_LEN = trial.suggest_categorical("max_len", [100, 128]) 
    #MAX_LEN = 128
    
    n_classes = 3 
    
    ### Define data loaders
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, batch_size, shuffle = True)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, batch_size, shuffle = False)
    
    ### Create an instance of the model
    model = SentimentClassifier(n_classes, dropout_prob)
    model = model.to(device)
    
    ### Define the number of warmup steps
    total_steps = len(train_data_loader)*num_epochs
    num_warmup_steps = total_steps * percentage
    
    ### Define the criterion, optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    ### Training loop
    for epoch in range(num_epochs):
        model.train()
        for d in train_data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            ### Forward propagation
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
                   
            ### Backpropagation step 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ### Used to avoid gradient explotion
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            

        ### Validation loop
        val_losses = []
        model.eval()
        with torch.no_grad():
            for d in val_data_loader:      
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                targets = d['targets'].to(device)
                
                ### Forward propagation
                outputs = model(input_ids, attention_mask)
                val_loss = loss_fn(outputs, targets)
                val_losses.append(val_loss.item())

    return np.mean(val_losses)
    
##########################################################################
#       Create a study object and optimize the objective function        #
##########################################################################

### Minimize the objective function
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=6)


##########################################################################
#                Display the results of the optimization                 #
##########################################################################

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")

for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    




  




  
  
  
  