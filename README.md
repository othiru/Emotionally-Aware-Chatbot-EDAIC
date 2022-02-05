# Emotionally Aware Chatbot EDAIC
## Table of contents
* [General info](#general-info)
* [Author](#author)
* [Supervisor](#supervisor)
* [Technologies](#technologies)
* [Dataset](#dataset)
* [Models](#models)
* [Result](#result)
* [Conclusion](#conclusion)

## General info 
A chatbot is a software application that uses Artificial Intelligence and generates human-like
conversations. In this research, EDAIC (Emotion Detector & Artificial Intellegent Chatbot) is a chatbot capable of textual interactions and detecting
emotions from the text. It can work as an emotion detector. EDAIC will answer the question
of a user after analyzing that question’s keywords. Many apps use this kind of chatbot in the
FAQ and Help sections so that users can converse with assigned chatbots and find answers
to their queries. Instead of interacting with real humans, users converse with a human-like
chatbot that acts like an actual human. In this case, chatbots are very helpful that can assist
in different applications and websites.

Artificial Intelligence is the latest technology. Creating a chatbot using this technology can
make anyone’s professional profile stand out in this competitive era. As Python language is
used to create this chatbot, learning this language can also enhance anyone’s professional
profile.

EDAIC, a chatbot that uses Artificial Intelligence to detect emotions from text, is introduced
in this research. Emotions refer to various types of consciousness or states of mind that are
shown as feelings. They can be expressed through facial expressions, gestures, texts, and
speeches. Emotion detection through chatbots also helps to understand humans more precisely
while conversing with them. Emotions like Joy, Love, Surprise, Neutral, Sadness, Fear
and Anger are considered while creating this chatbot. This chatbot will communicate with
people using the help of natural language processing (NLP) and classification algorithms.

In this research, EDAIC is more likely to work as an emotion detector. By detecting emotions,
it can understand the customers’ likes or dislikes for the products and services of a company
more accurately. It can assist users with swift responses that will reduce time and carry out
the service faster. Therefore, this chatbot helps to reduce the workload. It aids the business
teams in communicating with customers to resolve their queries in a faster and optimized
way.


## Author
* Nowshin Rumali
* Amin Ahmed Toshib
* Rejone E Rasul Hridoy
* Mehedi Hasan Sami


## Supervisor
* **Mr. Md. Khairul Hasan**

## Technologies
Project is created with:
* **Google Colab**
* **Pycharm**

Language used in this project is:
* **Python**
	
## Dataset

* **Chatbot Dataset**: There are 2064 data in this dataset and 3 columns: User text, Chatbot reply and intent. This dataset is merged from 2 datasets and contains 27 unique intents.
* **Tweet Emotion Dataset**: Tweet emotions from SemEval-2018 Affect in Tweets Distant Supervision Corpus (AIT-2018 Dataset) is used.This dataset has two columns content and sentiment. It has 25000 unique text classified with 7 emotions such as anger, love, surprise, fear, joy, sadness and Neutral.

Both Dataset are given in this repository

## Models
Here we have used 7 machine learning models for both chatbot and emotion detector and the models are:
* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* Multinomial Naive Bayes
* Decision Tree Classifier
* Multilayer Perceptron (MLP)

## Result
1. **Chatbot**: Accuracy of the seven models - Support Vector Machine (SVM), Logistic Regression, Random
Forest Classifier, XGBoost Classifier, Multinomial Naive Bayes, Decision Tree Classifier and Multilayer Perceptron
are `71.71%`, `56.01%`, `48.45%`, `58.72%`, `44.19%`, `53.68%` and `70.93%` respectively.
2. **Emotion Detector**: Accuracy of the seven models - Support Vector Machine (SVM), Logistic Regression, Random
Forest Classifier, XGBoost Classifier, Multinomial Naive Bayes, Decision Tree Classifier and Multilayer Perceptron
are `88.47%`, `86.12%`, `66.87%`, `88.84%`, `71.59%`, `75.75%` and `85.31%` respectively.

## Conclusion 
In our model, we have developed an application that has created an atmosphere where our
chatbot (EDAIC) and a human can make a conversation. We have applied seven machine
learning models to our chatbot. They are SVM, Logistic Regression, Random Forest, XGBoost,
Naive Bayes, Decision Tree, and Multilayer Perceptron. We have generated the reply
based on the best intent and cosine distance. At first, We have used ensemble learning to
choose the best intent. After finding the best intent, we have used cosine distance to receive
relatively suitable responses. Among all the models, SVM has given the most accuracy for
chats that is `71.7%`.

Detecting emotions from text is relatively complex. Because, in some cases, it is not easy to
recognize appropriate emotions from the text. Again, we have used seven machine learning
models for our emotion detector, and XGBoost gave the most accuracy that is `88.84%`.


We have implemented an application combining two models - chatbot and emotion detector.
Here, users can chat with EDAIC, and they can also see the emotions of their texts. There
is a feature for rating emotions. So, one can easily give a rating to the emotion that has
been predicted by the chatbot. Also, a user can choose the correct emotion if the predicted
emotion seems incorrect to the user and gives a bad rating. If the predicted emotion is
rated by a user, this feedback will be stored in a file. So, after collecting a certain amount
of feedback we can re-train our model for better performance.



