
<div align="center">

![logo](https://github.com/ShailadhShinde/Quora/blob/main/assets/header.png)  
<h1 align="center"><strong>Quora Insincere Questions Classification

  <h6 align="center">Detect toxic content to improve online conversations (CV) project</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)

</div>
This project focuses on:

- Exploratory Data Analysis
- 
-
-

#### -- Project Status: [Completed]

#### -- time_series.py / time_Series.ipynb - Contains code for the project

#### -- eda-time-series.ipynb / EDA.py - Exploratory Data Analysis [Click to view](https://www.kaggle.com/code/shailadh/eda-time-series?scriptVersionId=190759981)

----

## [📚 Project Documentation 📚](http://smp.readthedocs.io/)

### 📋 Table of Contents
- [Overview](#overview)
  - [About the dataset](#atd)
  - [Sample Selection](#ss)
  - [Preprocessing](#pp)
  - [Feature Engineering](#fe)
  - [Evaluation](#eval)
  - [Model](#model)
- [Results](#results)
- [Getting Started](#gs)
  - [Prerequisites](#pr)


###  📌 Project Overview  <a name="overview"></a>

This project is a Notebook about a Classification problem . An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.  On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
** The purpose is to develop models that identify these insincere questions. **

- ### About the dataset  <a name="atd"></a>

`The training data `includes the question that was asked, and whether it was identified as insincere (target = 1). The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.

- ### Embeddings
    External data sources are not allowed . Although, they have provided a number of word embeddings along with the dataset that can be used in the models. These are as follows:

   - GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
   - glove.840B.300d - https://nlp.stanford.edu/projects/glove/
   - paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
   - wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html
     
- ### Sample Selection  <a name="ss"></a>

  Used 2017 data to extract and construct samples

  train date: 20170228 - 20170830

  validation: 20170725 - 20170808
  
- ### Preprocessing  <a name="pp"></a>
    - Used all tokens from both train and test data for our vocabulary
    - Split by space afterwards using spacy (Spacy tokenizer)
    - Used glove for embeddings
    - No truncation of tokens
    - Tried stemmer, lemmatizer, spell correcter, etc. to find word vectors
    -  Local solid CV to tune all the hyperparameters
    
- ### Model Structure  <a name="fe"></a>

 
  
- ### Evaluation  <a name="eval"></a>
  The evaluation metric used is Root Mean Squared Logarithmic Error. RMSLE = $\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$

- ### Model <a name="model"></a>
    16 Light Gradient Boosting Model trained for each day

----

## 💫 Results <a name="results"></a>

  Got a good score resulting in top 1 % of the kaggle leader board
  
   <p align="center">
  <img width="60%" src="https://github.com/ShailadhShinde/Time_series/blob/main/assets/score.JPG">
 </p>

  
---

## 🚀 Getting Started <a name="gs"></a>

### ✅ Prerequisites <a name="pr"></a>
 
 - <b>Dataset prerequisite for training</b>:
 
 Before starting to train a model, make sure to download the dataset from <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data" target="_blank">here </a> or add it to your notebook
 ### 🐳 Setting up and Running the project

 Just download/copy the files `time_series.py / time_Series.ipynb ` and `EDA.ipynb / EDA.py ` and run them

  
