print('-----------------------------------------------------------------------')
print("Hello World")
print('-----------------------------------------------------------------------')

print("GPU IS AVAILABLE", torch.cuda.is_available())

### Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##########################################################################
#                        Importing dependencies                          #
##########################################################################

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

##########################################################################
#         Visualizations to help estimate the model's performance        #
##########################################################################

### Plot the ROC curve 
def multiclass_roc_curve(y_true, y_score, n_classes):
    """
    Compute ROC curve and ROC area for each class in a multiclass classification setting.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores (predicted probabilities) of the positive class for each sample.

    n_classes : int
        Number of classes in the classification problem.

    Returns
    -------
    dict
        A dictionary of ROC curves and ROC AUC scores for each class.
    """
    y_true = label_binarize(y_true, classes=[0, 1, 2])
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        ### fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    ### Plot ROC curve for each class
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(
                  fpr[i], 
                  tpr[i], 
                  color=color, 
                  lw=lw, 
                  label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
        )
    ### Plot micro-averaged ROC curve
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        color="deeppink",
        linestyle=":",
        linewidth=4,
        label="Micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    )        
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic for multi-class data')
    ax.legend(loc="lower right")
    fig.savefig('BERT_results/roc.png')

    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
    
    
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

### Plot the confusion matrix
def show_confusion_matrix(confusion_matrix):
  fig, ax = plt.subplots(figsize=(8,8))
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  ax.set_ylabel('True sentiment')
  ax.set_xlabel('Predicted sentiment')
  fig.savefig("BERT_results/confusion_matrix.png")

### Plot the Precision-Recall curve
def multiclass_precision_recall_curve(y_true, y_score, n_classes):
    """
    Compute Precision-Recall curve for each class in a multiclass classification setting.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores (predicted probabilities) of the positive class for each sample.

    n_classes : int
        Number of classes in the classification problem.

    Returns
    -------
    dict
        A dictionary of Precision-Recall curves and average Precision scores for each class.
    """
    # Check input shapes and data types
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    assert y_true.ndim == 1
    assert y_score.ndim == 2
    assert y_true.shape[0] == y_score.shape[0]
    assert y_score.shape[1] == n_classes

    precision = dict()
    recall = dict()
    average_precision = dict()

    ### Compute Precision-Recall curve and average Precision score for each class
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_score[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    ### Plot Precision-Recall curve for each class
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    colors = ["blue", "red", "green"]
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label="Precision-Recall curve of class {0} (AP = {1:0.2f})".format(i, average_precision[i]),
        )

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve for multi-class data")
    ax.legend(loc="best")
    fig.savefig("BERT_results/pr.png")

    return {"precision": precision, "recall": recall, "average_precision": average_precision}

    
RANDOM_SEED = 0 #42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

### Read .csv file
df_all = pd.read_csv("/home/pfc/atoval/datasets/final_df_preprocessed.csv", encoding='utf-8')
df_neutrals = pd.read_csv("/home/pfc/atoval/datasets/final_df_neutrals_preprocessed.csv", encoding='utf-8')

### Concatenate both dataframes
df = pd.concat([df_all, df_neutrals])


### Transform column 'sentiment' to integers
df['sentiment'] = df['sentiment'].astype(int)
df = df.drop_duplicates().dropna()

##########################################################################################################

### Define number of samples per class
#n_samples = 100

### Group data by class and sample n_samples from each group
#df = df.groupby('sentiment').apply(lambda x: x.sample(n_samples)).reset_index(drop=True)

##########################################################################################################

### Count the values in the 'sentiment' column
value_counts = df['sentiment'].value_counts()
print("Value counts in column 'label':\n", value_counts)

### Drop rows with neutral sentiment
#df = df[df['sentiment'] != 2]

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

##########################################################################
#           Obtain the maximum length of the token sentences             #
##########################################################################

### Encode our concatenated data
encoded_tweets = [tokenizer.encode(tweet, add_special_tokens=True) for tweet in df.preprocessed_tweet]

### Find the maximum length
lens = [len(tweet) for tweet in encoded_tweets]

print('Max length: ', max(lens))
print('Mean length: ', np.mean(lens))

MAX_LEN = 128

print('LENGTH: ', MAX_LEN)

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

##########################################################################
#                  Define the hyperparameters to use                     #
##########################################################################

BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS = 3
DROPOUT_PROB = 0.5
WEIGHT_DECAY = 1e-8
LEARNING_RATE = 2e-5

print("BATCH_SIZE: ",BATCH_SIZE)
print("EPOCHS: ",EPOCHS)
print("DROPOUT_PROB: ",DROPOUT_PROB)
print("LR: ",LEARNING_RATE)

df_train, df_val = train_test_split(df, test_size = 0.1, random_state = RANDOM_SEED, shuffle = True)
df_val, df_test = train_test_split(df_val, test_size = 0.5, random_state = RANDOM_SEED, shuffle = True)

print(df_train.shape, df_val.shape, df_test.shape)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, shuffle = True)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE, shuffle = False)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE, shuffle = False)

### Get one batch as example.
sample_batch = next(iter(test_data_loader))
# print(sample_batch['input_ids'].shape,sample_batch['attention_mask'].shape,sample_batch['targets'].shape) 

print("Created w/ success")


##########################################################################
#                        Define the BERT model                           #
##########################################################################

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    ### Freeze the first 10 layers of the BERT
    
    #for name, param in self.bert.named_parameters():
        #if ( not name.startswith('pooler')) and "layer.11" not in name and "layer.10" not in name:
            #param.requires_grad = False
            
    self.drop = nn.Dropout(p = DROPOUT_PROB)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    pooled_output = self.bert(
      input_ids = input_ids,
      attention_mask = attention_mask
    )['pooler_output']

    output = self.drop(pooled_output)
    return self.out(output)
    
    
### Create an instance of the model
model = SentimentClassifier(NUM_CLASSES)
model = model.to(device)

### Define the optimizer, scheduler and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, eps = WEIGHT_DECAY)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = total_steps * 0.1,
    num_training_steps = total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


##########################################################################
#                    Function to train the model                         #
##########################################################################

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  
  model = model.train()

  train_losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d['input_ids'].to(device)
    attention_mask = d['attention_mask'].to(device)
    targets = d['targets'].to(device)
    
    ### Forward propagation
    outputs = model(input_ids, attention_mask)
    loss = loss_fn(outputs, targets)
    _, preds = torch.max(outputs, dim=1)
    
    ### Append training loss and get number of correct predictions
    correct_predictions += torch.sum(preds == targets)
    train_losses.append(loss.item())

    ### Backpropagation step 
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ### Used to avoid gradient explotion
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(train_losses)
  
  
##########################################################################
#                   Function to evaluate the model                       #
##########################################################################
 
def eval_model(model, data_loader, loss_fn, device, n_examples):
  
  model = model.eval()

  val_losses = []
  correct_predictions = 0

  ### Not necessary to calculate the gradient on evaluation
  with torch.no_grad(): 
    for d in data_loader:
      input_ids = d['input_ids'].to(device)
      attention_mask = d['attention_mask'].to(device)
      targets = d['targets'].to(device)

      ### Forward propagation
      outputs = model(input_ids, attention_mask)
      loss = loss_fn(outputs, targets)
      _, preds = torch.max(outputs, dim=1)

      correct_predictions += torch.sum(preds == targets)
      val_losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(val_losses)


##########################################################################
#             Function to get predictions from the model                 #
##########################################################################

def get_predictions(model, data_loader):
  
  model = model.eval()
  
  predictions = []
  prediction_probs = []
  real_values = []
  tweets_text = []

  with torch.no_grad():
    for d in data_loader:
      tweets = d['tweet']
      input_ids = d['input_ids'].to(device)
      attention_mask = d['attention_mask'].to(device)
      targets = d['targets'].to(device)

      outputs = model(
        input_ids = input_ids,
        attention_mask = attention_mask
      )

      _, preds = torch.max(outputs, dim=1)
      
      ### Output as probabilities
      probs = F.softmax(outputs, dim=1)

      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)
      tweets_text.append(tweets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  
  return tweets_text, predictions, prediction_probs, real_values

  
##########################################################################
#                   Train and validate the the model                     #
##########################################################################

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 150)
    
    train_acc, train_loss = train_model(
            model, 
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        
    )
    
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )   
    
    print(f'Train loss: {train_loss}, Train accuracy: {train_acc}, Validation loss: {val_loss}, Validation accuracy: {val_acc}')
    
    history['train_acc'].append(train_acc.item())
    history['train_loss'].append(train_loss.item())
    history['val_acc'].append(val_acc.item())
    history['val_loss'].append(val_loss.item())
    
    if val_acc > best_accuracy:
        ### Save the model state dictionary to a file
        torch.save(model.state_dict(), 'BERT_results/best_model_state.bin')
        best_accuracy = val_acc

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(history['train_acc'], label='train accuracy')
ax.plot(history['val_acc'], label='validation accuracy')
ax.set_title('Training history')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend()

ax.set_xticks(range(EPOCHS))
ax.set_ylim([0, 1])
fig.savefig("BERT_results/train_VS_val.png");

print('-' * 150)      
print('Best validation accuracy:', best_accuracy.item())
print('-' * 150)
print('All history \n', history)


##########################################################################
#                         Evaluating the the model                       #
##########################################################################
  
test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,    
        device,
        len(df_test)
)

print("Test accuracy", test_acc.item())
  
### Similar to evaluation function, but we're storing the actual tweet, the predicted class, and the predicted probability for each class
tweets_text, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

print("CONFUSION MATRIX \n")
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=['positive', 'negative', 'neutral'], columns=['positive', 'negative', 'neutral'])
show_confusion_matrix(df_cm);

### Classification Report
print(classification_report(y_test, y_pred, target_names=['positive', 'negative', 'neutral']));

print("ROC CURVE \n")
multiclass_roc_curve(y_test.numpy(), y_pred_probs.numpy(), NUM_CLASSES);

print("PR CURVE \n")
multiclass_precision_recall_curve(y_test.numpy(), y_pred_probs.numpy(), NUM_CLASSES);

print("---------------------------------- EXAMPLE --------------------------------------")

example = "i hate my life, this sucks a lot"

def get_sentiment(text):
    
    targets = ['positive', 'negative', 'neutral']
    
    encoded_text = tokenizer.encode_plus(
              text,
              add_special_tokens     = True,
              max_length             = MAX_LEN,
              return_token_type_ids  = False,
              return_attention_mask  = True,
              return_tensors         = 'pt',
              padding                = 'max_length',
              truncation             = True
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    ### Prediction output
    output = model(input_ids, attention_mask)
    _, preds = torch.max(output, dim=1)
    
    ### Output as probabilities 
    probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
    print(probs)

    print(f'positive % : {probs[0]:.4f}, negative % : {probs[1]:.4f}, neutral % : {probs[2]:.4f}')
    print(f'Sentiment: {targets[preds]}')
    
get_sentiment(example)


  
