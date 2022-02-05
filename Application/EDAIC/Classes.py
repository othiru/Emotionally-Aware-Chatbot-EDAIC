import pandas as pd
import nltk
import re
import string

# Feature Extraction Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

# Classifier Model libraries
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

# Performance Matrix libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# other
import pickle
import os
import random
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")

dataset_folder = 'Datasets/'
saved_model_folder = 'SavedModels/'

### --------------------------Preprocessing-------------------------------
class Preprocessing:
    def __init__(self, Remove_stopwords=True):
        self.emojis = pd.read_csv(dataset_folder + 'emojis.txt', sep=',', header=None)
        self.emojis_dict = {i: j for i, j in zip(self.emojis[0], self.emojis[1])}
        self.pattern = '|'.join(sorted(re.escape(k) for k in self.emojis_dict))
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        self.rmv_stopword = Remove_stopwords

    def replace_emojis(self, text):
        text = re.sub(self.pattern, lambda m: self.emojis_dict.get(m.group(0)), text, flags=re.IGNORECASE)
        return text

    def remove_punct(self, text):
        text = self.replace_emojis(text)
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text

    def tokenization(self, text):
        text = text.lower()
        text = re.split('\W+', text)
        return text

    def remove_stopwords(self, text):
        stopword = nltk.corpus.stopwords.words('english')
        stopword.extend(
            ['yr', 'year', 'woman', 'man', 'girl', 'boy', 'one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
             'treatment', 'associated', 'patients', 'may', 'day', 'case', 'old', 'u', 'n', 'didnt', 'ive', 'ate',
             'feel', 'keep'
                , 'brother', 'dad', 'basic', 'im', ''])

        text = [word for word in text if word not in stopword]
        return text

    def lemmatizer(self, text):
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in text]
        return text

    def clean_text(self, text):
        text = self.remove_punct(text)
        text = self.tokenization(text)
        if self.rmv_stopword == True:
            text = self.remove_stopwords(text)
        text = self.lemmatizer(text)
        return text


### ----------------Feature Extraction-------------------

class FeatureExtraction:
    def __init__(self,rmv_stopword=True):
        self.rmv_stopword = rmv_stopword
        self.preprocess = Preprocessing(self.rmv_stopword)
        self.countVectorizer1 = CountVectorizer(analyzer=self.preprocess.clean_text)
        self.tfidf_transformer_xtrain = TfidfTransformer()
        self.tfidf_transformer_xtest = TfidfTransformer()

    def get_features(self, X_train, X_test):
        # countVectorizer1 = CountVectorizer(analyzer=self.preprocess.clean_text)
        countVector1 = self.countVectorizer1.fit_transform(X_train)

        countVector2 = self.countVectorizer1.transform(X_test)

        # tfidf_transformer_xtrain = TfidfTransformer()
        x_train = self.tfidf_transformer_xtrain.fit_transform(countVector1)

        # tfidf_transformer_xtest = TfidfTransformer()
        x_test = self.tfidf_transformer_xtest.fit_transform(countVector2)

        return x_train, x_test

    def get_processed_text(self, input_str):
        return self.tfidf_transformer_xtest.fit_transform(self.countVectorizer1.transform([input_str]))

### ---------------Models------------------------------------------------
class Models:
    def __init__(self, X_train, Y_train, X_test, Y_test, model_name='cb'):
        self.x_train = X_train
        self.x_test = X_test
        self.y_test = Y_test
        self.y_train = Y_train
        self.chatbot_model_file = saved_model_folder + 'Chatbot Models_7 models.pkl'
        self.emotion_model_file = saved_model_folder + 'Emotion Detection Models_7 models.pkl'

        self.chatbot_summary_file = saved_model_folder + 'Chatbot Models Summary_7 models.pkl'
        self.emotion_summary_file = saved_model_folder + 'Emotion Detection Models Summary_7 models.pkl'
        self.model_name = model_name  # cb = chatbot model, ed = emotion detection model

        self.svm = SGDClassifier()
        self.logisticRegr = LogisticRegression()
        self.rfc = RandomForestClassifier(n_estimators=1, random_state=0)
        self.xgbc = XGBClassifier(max_depth=16, n_estimators=1000, nthread=6)
        self.mnb = MultinomialNB()
        self.dt = tree.DecisionTreeClassifier()
        self.mlp = MLPClassifier(random_state=5, max_iter=300)

        self.svm_summary = {}
        self.lr_summary = {}
        self.rfc_summary = {}
        self.xgbc_summary = {}
        self.mnb_summary = {}
        self.dt_summary = {}
        self.mlp_summary = {}

    def load_models(self):
        if self.model_name == 'ed':
            if os.path.isfile(self.emotion_model_file):
                with open(self.emotion_model_file, 'rb') as f:
                    self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp = pickle.load(f)

                with open(self.emotion_summary_file, 'rb') as f:
                    self.svm_summary, self.lr_summary, self.rfc_summary, self.xgbc_summary, self.mnb_summary, self.dt_summary, self.mlp_summary = pickle.load(
                        f)
                    print('Emotion Detection Models retrived from Disk successfully')
                    return self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp
            else:
                self.train_models()
                self.save_models()
                return self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp
        elif self.model_name == 'cb':
            if os.path.isfile(self.chatbot_model_file):
                with open(self.chatbot_model_file, 'rb') as f:
                    self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp = pickle.load(f)

                with open(self.chatbot_summary_file, 'rb') as f:
                    self.svm_summary, self.lr_summary, self.rfc_summary, self.xgbc_summary, self.mnb_summary, self.dt_summary, self.mlp_summary = pickle.load(
                        f)
                    print('Chabot Models retrived from Disk successfully')
                    return self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp
            else:
                self.train_models()
                self.save_models()
                return self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp

    def train_models(self):
        print('-----Model Training-----')
        print('Training SVM...')
        self.SVM()
        print('Training Logistic Regression...')
        self.LR()
        print('Training Random Forest...')
        self.RFC()
        print('Training XGBoost...')
        self.XGBC()
        print('Training Multinomial Naive Bayes...')
        self.MNB()
        print('Training Decision Tree...')
        self.DT()
        print('Training Multi-Layer Perceptron Model...')
        self.MLP()
        print('Successfully Trained All Models')

        return self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp

    def SVM(self):
        self.svm.fit(self.x_train, self.y_train)
        y_pred = self.svm.predict(self.x_test)

        svm_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        svm_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        svm_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        svm_cm = confusion_matrix(self.y_test, y_pred)
        svm_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.svm_summary['Accuracy'] = svm_acc
        self.svm_summary['Precision'] = svm_prec
        self.svm_summary['Recall'] = svm_recal
        self.svm_summary['F1'] = svm_f1
        self.svm_summary['CM'] = svm_cm

    def LR(self):
        self.logisticRegr.fit(self.x_train, self.y_train)

        y_pred = self.logisticRegr.predict(self.x_test)

        lr_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        lr_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        lr_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        lr_cm = confusion_matrix(self.y_test, y_pred)
        lr_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.lr_summary['Accuracy'] = lr_acc
        self.lr_summary['Precision'] = lr_prec
        self.lr_summary['Recall'] = lr_recal
        self.lr_summary['F1'] = lr_f1
        self.lr_summary['CM'] = lr_cm

    def RFC(self):
        self.rfc.fit(self.x_train, self.y_train)

        y_pred = self.rfc.predict(self.x_test)

        rfc_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        rfc_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        rfc_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        rfc_cm = confusion_matrix(self.y_test, y_pred)
        rfc_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.rfc_summary['Accuracy'] = rfc_acc
        self.rfc_summary['Precision'] = rfc_prec
        self.rfc_summary['Recall'] = rfc_recal
        self.rfc_summary['F1'] = rfc_f1
        self.rfc_summary['CM'] = rfc_cm

    def XGBC(self):
        self.xgbc.fit(self.x_train, self.y_train)
        y_pred = self.xgbc.predict(self.x_test)

        xgbc_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        xgbc_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        xgbc_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        xgbc_cm = confusion_matrix(self.y_test, y_pred)
        xgbc_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.xgbc_summary['Accuracy'] = xgbc_acc
        self.xgbc_summary['Precision'] = xgbc_prec
        self.xgbc_summary['Recall'] = xgbc_recal
        self.xgbc_summary['F1'] = xgbc_f1
        self.xgbc_summary['CM'] = xgbc_cm

    def MNB(self):
        self.mnb.fit(self.x_train, self.y_train)

        y_pred = self.mnb.predict(self.x_test)

        mnb_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        mnb_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        mnb_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        mnb_cm = confusion_matrix(self.y_test, y_pred)
        mnb_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.mnb_summary['Accuracy'] = mnb_acc
        self.mnb_summary['Precision'] = mnb_prec
        self.mnb_summary['Recall'] = mnb_recal
        self.mnb_summary['F1'] = mnb_f1
        self.mnb_summary['CM'] = mnb_cm

    def DT(self):
        self.dt.fit(self.x_train, self.y_train)
        y_pred = self.dt.predict(self.x_test)

        dt_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        dt_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        dt_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        dt_cm = confusion_matrix(self.y_test, y_pred)
        dt_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.dt_summary['Accuracy'] = dt_acc
        self.dt_summary['Precision'] = dt_prec
        self.dt_summary['Recall'] = dt_recal
        self.dt_summary['F1'] = dt_f1
        self.dt_summary['CM'] = dt_cm

    def MLP(self):
        self.mlp.fit(self.x_train, self.y_train)
        y_pred = self.mlp.predict(self.x_test)

        mlp_acc = round(accuracy_score(y_pred, self.y_test) * 100, 3)
        mlp_prec = round(precision_score(self.y_test, y_pred, average='macro') * 100, 3)
        mlp_recal = round(recall_score(self.y_test, y_pred, average='macro') * 100, 3)
        mlp_cm = confusion_matrix(self.y_test, y_pred)
        mlp_f1 = round(f1_score(self.y_test, y_pred, average='macro') * 100, 3)
        self.mlp_summary['Accuracy'] = mlp_acc
        self.mlp_summary['Precision'] = mlp_prec
        self.mlp_summary['Recall'] = mlp_recal
        self.mlp_summary['F1'] = mlp_f1
        self.mlp_summary['CM'] = mlp_cm

    def model_summary(self):
        return self.svm_summary, self.lr_summary, self.rfc_summary, self.xgbc_summary, self.mnb_summary, self.dt_summary, self.mlp_summary

    def save_models(self):
        if self.model_name == 'ed':
            with open(self.emotion_model_file, 'wb') as f:
                pickle.dump([self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp], f)

            with open(self.emotion_summary_file, 'wb') as f:
                pickle.dump([self.svm_summary, self.lr_summary, self.rfc_summary, self.xgbc_summary, self.mnb_summary,
                             self.dt_summary, self.mlp_summary], f)

            print('Emotion Detection Models saved successfully in the disk')
        elif self.model_name == 'cb':
            with open(self.chatbot_model_file, 'wb') as f:
                pickle.dump([self.svm, self.logisticRegr, self.rfc, self.xgbc, self.mnb, self.dt, self.mlp], f)

            with open(self.chatbot_summary_file, 'wb') as f:
                pickle.dump([self.svm_summary, self.lr_summary, self.rfc_summary, self.xgbc_summary, self.mnb_summary,
                             self.dt_summary, self.mlp_summary], f)

            print('Chatbot Models saved successfully in the disk')




