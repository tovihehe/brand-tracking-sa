# Brand Sentiment Analysis Using BERT and RoBERTa 💖

## Abstract

Brand sentiment monitoring is crucial for businesses to understand customer perceptions. Real-time analysis of social media data presents challenges, but advanced transformer-based models offer robust solutions. This project focuses on fine-tuning two prominent transformer models, BERT and RoBERTa, to achieve state-of-the-art sentiment analysis results. A user-friendly interface developed with Streamlit will showcase sentiment analysis outcomes, enabling companies to gauge public sentiment towards their products and services effectively.

## Introduction

In today's competitive market, understanding brand sentiment is essential for businesses to maintain a positive brand identity and respond proactively to potential issues. This project aims to demonstrate how advanced models can be utilized to analyze sentiment in real-time, leveraging Twitter data to offer insights into public perception of brands.

## State-of-the-Art

### Transformer Models

The transformer architecture, introduced by Google in 2017, revolutionized Natural Language Processing (NLP). Models based on transformers, such as BERT and RoBERTa, have set new standards for NLP tasks by effectively capturing contextual relationships and long-term dependencies in text.

- **BERT (Bidirectional Encoder Representations from Transformers)**: BERT captures context from both directions (left-to-right and right-to-left) in a bidirectional manner. It uses a masked language model and next-sentence prediction to learn deep contextual information.

- **RoBERTa (Robustly Optimized BERT Approach)**: RoBERTa builds on BERT's architecture but optimizes several aspects, such as using dynamic masking and training with larger mini-batches. It excludes the next-sentence prediction task and increases the amount of training data and duration, resulting in improved performance over BERT.

## Objectives

1. Fine-tune BERT and RoBERTa models for sentiment analysis using a dataset of labeled tweets.
2. Compare the performance of both models.
3. Scrape tweets from Twitter and analyze their sentiment.
4. Visualize sentiment analysis results in real-time using a Streamlit-based dashboard.

## Data Collection and Preparation

### Data Collection

To train our models effectively, a large dataset of labeled tweets was compiled from sources such as Apple product reviews and well-known datasets like Sentiment140 and SemEval. Tweets were pre-processed to remove irrelevant content and ensure quality data.

### Data Augmentation

To address class imbalance, particularly in neutral sentiment instances, back-translation was used to augment the dataset. This technique involves translating tweets to another language and then back to the original language to generate variations of the original text, enhancing the dataset's diversity.

## Model Overview

### BERT and RoBERTa Architectures

Both BERT and RoBERTa utilize transformer encoder layers with multi-head self-attention mechanisms. They generate contextually rich embeddings that capture nuanced word relationships. The main differences between BERT and RoBERTa lie in their pre-training strategies and hyperparameter configurations.

### Pre-training and Fine-tuning

- **Pre-training**: Both models were pre-trained on large text corpora. BERT uses masked language modeling and next-sentence prediction, while RoBERTa employs dynamic masking and trains with larger datasets and higher learning rates.

- **Fine-tuning**: Fine-tuning involves adapting pre-trained models to specific tasks by adding task-specific layers and adjusting hyperparameters. For sentiment analysis, a classification layer was added on top of the model outputs.

## Hyperparameter Optimization

### Learning Rate Finder and Optuna

Hyperparameter tuning was conducted using the Learning Rate Finder to identify an optimal learning rate. Optuna was employed for broader hyperparameter optimization, adjusting parameters such as batch size and dropout rates to enhance model performance.

## Results

The sentiment analysis dataset comprised 1,764,293 instances. Models were evaluated using precision, recall, F1-score, and accuracy metrics. Both BERT and RoBERTa demonstrated high performance, with RoBERTa slightly outperforming BERT in certain metrics. Fine-tuning the entire model yielded better results than selectively unfreezing specific layers.

## Dashboard Demo

### Twitter API for Data Scraping

Tweets were scraped using the Twitter API, with considerations for recent policy changes and pricing. Data was stored in an SQLite3 database for efficient management and analysis.

### Streamlit Interface

A Streamlit-based dashboard was developed to display sentiment analysis results. The interface allows users to query sentiment for specific brands, visualize trends, and interact with the data in real-time.


<div style="display: inline-flex; align-items: center;">
    <img src="https://github.com/user-attachments/assets/1a7d39dc-98c4-450d-a96f-5d37170b1d38" alt="Description of Image 1" width="400"/>
    <img src="https://github.com/user-attachments/assets/502d463e-9879-4047-bd46-0d56d70c12bb" alt="Description of Image 2" width="400"/>
</div>

## Conclusion

This project demonstrated the effective use of BERT and RoBERTa for sentiment analysis, highlighting the importance of advanced NLP models in real-time brand monitoring. The Streamlit dashboard offers a practical solution for businesses to assess public sentiment and make informed decisions.

## References

1. [Brand Sentiment Analysis](https://www.explorium.ai/blog/business-data/brand-sentiment-data/)
2. Vaswani et al. "Attention Is All You Need." 2017. [Link](https://arxiv.org/abs/1706.03762) 
3. Sentiment140 Dataset. [Link](https://www.kaggle.com/datasets/kazanova/sentiment140) 
4. SemEval Dataset. [Link](https://www.kaggle.com/datasets/azzouza2018/semevaldatadets)
5. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." 2018. [Link](https://arxiv.org/abs/1810.04805)
6. Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." 2019. [Link](https://arxiv.org/abs/1907.11692)
7. BERT Tokenization. [Link](https://huggingface.co/transformers/tokenizer_summary.html)
8. BERT Fine-Tuning. [Link](https://github.com/google-research/bert)
9. Optuna Documentation. [Link](https://optuna.org/)
10. LRFinder Documentation. [Link](https://docs.fast.ai/callbacks.lrfinder.html)
11. Leslie Smith. "Cyclical Learning Rates for Training Neural Networks." 2015. [Link](https://arxiv.org/abs/1506.01186)
12. Sklearn Metrics. [Link](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
13. Twitter API Pricing. [Link](https://developer.twitter.com/en/pricing)
14. SQLite3 Documentation. [Link](https://www.sqlite.org/docs.html)
15. Streamlit Documentation. [Link](https://streamlit.io/) 
