
<div align="center">

![logo](https://github.com/ShailadhShinde/Quora/blob/main/assets/header.png)  
<h1 align="center"><strong>Quora Insincere Questions Classification

  <h6 align="center">Detect toxic content to improve online conversations (CV) project</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)

</div>
This project focuses on:

- Implementation of recurrent neural network layers like BiLSTM, GRU
- Using word embeddings for text classification
- Advance Preprocessing on text

#### -- Project Status: [Completed]

#### -- quora.py / quora.ipynb - Contains code for the project

----

## [📚 Project Documentation 📚](http://smp.readthedocs.io/)

### 📋 Table of Contents
- [Overview](#overview)
  - [About the dataset](#atd)
  - [Embeddings](#embed)
  - [Preprocessing](#pp)
  - [Model Structure](#ms)
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

- ### Embeddings <a name="embed"></a>

    External data sources are not allowed . Although, they have provided a number of word embeddings along with the dataset that can be used in the models. These are as follows:

   - GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
   - glove.840B.300d - https://nlp.stanford.edu/projects/glove/
   - paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
   - wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html
  
- ### Preprocessing  <a name="pp"></a>

    - Used all tokens from both train and test data for our vocabulary
    - Split by space afterwards using spacy (Spacy tokenizer)
    - Used glove for embeddings
    - No truncation of tokens
    - Tried stemmer, lemmatizer, spell correcter, etc. to find word vectors
    - Local CV to tune all the hyperparameters
    
- ### Model Structure  <a name="ms"></a>

 ![Model](https://github.com/ShailadhShinde/Quora/blob/main/assets/model.JPG)
  
- ### Evaluation  <a name="eval"></a>
  The evaluation metric used is F1 Score.
  ![F1 score](https://github.com/ShailadhShinde/Quora/blob/main/assets/fi.svg)

----

## 💫 Results <a name="results"></a>

  Got a good score resulting in top 1 % of the kaggle leader board
  
   <p align="center">
  <img width="60%" src="https://github.com/ShailadhShinde/Quora/blob/main/assets/score.JPG">
 </p>

  
---

## 🚀 Getting Started <a name="gs"></a>

### ✅ Prerequisites <a name="pr"></a>
 
 - <b>Dataset prerequisite for training</b>:
 
 Before starting to train a model, make sure to download the dataset from <a href="https://www.kaggle.com/competitions/quora-insincere-questions-classification/data" target="_blank">here </a> or add it to your notebook
 ### 🐳 Setting up and Running the project

 Just download/copy the files `quora.py / quora.ipynb ` and run them (Make sure to enable GPU)

  
