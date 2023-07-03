print('-----------------------------------------------------------------------')
print(":333 Hello World")
print('-----------------------------------------------------------------------')


#### Improve the optimization during the training, we are going to use a learning rate scheduler, this means we adjust the learning rate 
#### during the training loop either based on the number of epochs or validation measurements

### LR is one of the most important hyperparameters that you should twaek during the training
### By adjusting the learning rate, most of the times we want to decrease it, but it depends on the problem task

print("GPU IS AVAILABLE", torch.cuda.is_available())

### Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########################################################################
#                        Importing dependencies                          #
##########################################################################

import torch
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torch_lr_finder import LRFinder
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import FastaiLRFinder

### Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

### Read .csv file
df_all = pd.read_csv('datasets/final_df_preprocessed.csv', encoding='utf-8')
df_neutrals = pd.read_csv('datasets/final_df_neutrals_preprocessed.csv', encoding='utf-8')

### Concatenate both dataframes
df = pd.concat([df_all, df_neutrals])

### Transform column 'sentiment' to integers
df['sentiment'] = df['sentiment'].astype(int)
df = df.drop_duplicates().dropna()

##########################################################################################################

### Define number of samples per class
#n_samples = 20000

### Group data by class and sample n_samples from each group
#df = df.groupby('sentiment').apply(lambda x: x.sample(n_samples)).reset_index(drop=True)

##########################################################################################################

### Count the values in the 'sentiment' column
value_counts = df['sentiment'].value_counts()
print("Value counts in column 'label':\n", value_counts)

### Create TweetsDataset class
class TweetsDataset(Dataset):

  def __init__(self, tweets, targets):
    self.tweets    = tweets
    self.targets   = targets
  
  def __len__(self):
    return len(self.tweets)
  
  def __getitem__(self, item):
    tweet = str(self.tweets[item])
    target = torch.tensor(self.targets[item])

    return tweet, target
    
def split_dataset(dataset, train_ratio=0.9):
    len_train_set = int(len(dataset)*train_ratio)
    indices = list(range(len(dataset)))
    train_set = Subset(dataset, indices[:len_train_set])
    valid_set = Subset(dataset, indices[len_train_set:])
    return train_set, valid_set
    
    
ds = TweetsDataset(
    tweets    = df.preprocessed_tweet.to_numpy(),
    targets   = df.sentiment.to_numpy()
  )
  
train_set, valid_set = split_dataset(ds)

BATCH_SIZE = 16
train_data_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(valid_set, BATCH_SIZE)
  
print("Created w/ success")

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    ### self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    self.max_len = 128
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.drop = nn.Dropout(p = 0.5)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
                  
  def forward(self, tweets, labels=None):
    encoded_inputs = [self.tokenizer.encode_plus(
      tweet,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      return_attention_mask=True,
      return_tensors='pt',
      padding='max_length',
      truncation=True
    ) for tweet in tweets]
    
    logits_list = []
    for encoded_input in encoded_inputs:
      encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
      pooled_output = self.bert(**encoded_input)['pooler_output']
      output = self.drop(pooled_output)
      logits = self.out(output)
      logits_list.append(logits[0])
    
    return torch.stack(logits_list)
    

### Create an instance of the model
model = SentimentClassifier(n_classes=3)
model = model.to(device)

##########################################################################
#                       Define the LRFinder class                        #
##########################################################################

### Create the class MyLRFinder that inherits from LRFinder
class MyLRFinder(LRFinder):
    def _validate(self, val_iter, non_blocking_transfer=True):
        
        ### Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in val_iter:
                ### Move data to the device
                inputs, labels = self._move_to_device(
                    inputs, labels, non_blocking=non_blocking_transfer
                )

                ### Forward pass and loss computation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        return running_loss / len(val_iter.dataset)
       
##########################################################################
#             Define the wrapper class to compute the loss               #
##########################################################################
 
class LossWrapper(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, outputs, labels=None):

        if labels is not None:
            return self.loss_fn(outputs, labels)
        else:
            return outputs[0]
        
### Call the loss wrapper class and the optimizer
criterion = LossWrapper(nn.CrossEntropyLoss()) 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)


##########################################################################
#                     Obtain the best learning rate                      #
##########################################################################

print("NEW LR")

### Create a figure
fig, ax = plt.subplots()

### Create an instance of the LRFinder class
lr_finder = MyLRFinder(model, optimizer, criterion, device='cuda')
lr_finder.reset()

### Call the range_test method to start the search
lr_finder.range_test(train_data_loader, val_loader=val_data_loader, start_lr=1e-5, end_lr=1e-4, num_iter=10, step_mode="linear")

### Plot the loss vs. learning rate
lr_finder.plot(skip_start=0, skip_end=0, ax=ax)

### Save the figure
fig.savefig("lr_finder.png")




