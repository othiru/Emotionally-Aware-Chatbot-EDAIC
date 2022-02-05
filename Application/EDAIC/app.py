from tkinter import *
import tkinter as tk
import pandas as pd
import nltk
import numpy as np
import random
import csv
from datetime import datetime
import pendulum
from tkinter import messagebox

# Feature Extraction Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

import warnings
warnings.filterwarnings("ignore")

from Classes import Preprocessing, FeatureExtraction, Models

BG_GRAY = "#ABB2B9"
BG_COLOR = "#797EF6"
TEXT_COLOR_BLACK = "#000000"
TEXT_COLOR_WHITE = "#FFFFFF"
BUTTON_COLOR = "#34207E"
RADIO_BUTTON_COLOR = "#488AC7"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

dataset_folder = 'Datasets/'
saved_model_folder = 'SavedModels/'
desktop_path = 'C:/Users/user/Desktop/'


class Chatbot:
    def __init__(self):
        accuracies = np.array([svm_summary_cb['Accuracy'], lr_summary_cb['Accuracy'], rfc_summary_cb['Accuracy'],
                               xgbc_summary_cb['Accuracy'],
                               mnb_summary_cb['Accuracy'], dt_summary_cb['Accuracy'], mlp_summary_cb['Accuracy']])
        norm_accuracy = accuracies - min(accuracies)
        self.model_weight = norm_accuracy / sum(norm_accuracy)
        self.Intents = df_chatbot['Intent'].unique()
        self.Human_name = 'friend'

    def response_generate(self, text, intent_name):
        reply = self.respond(text, intent_name)
        return reply

    def cosine_distance_countvectorizer_method(self, s1, s2):
        # sentences to list
        allsentences = [s1, s2]

        # text to vector
        vectorizer = CountVectorizer()
        all_sentences_to_vector = vectorizer.fit_transform(allsentences)

        text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
        text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()

        # distance of similarity
        cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
        return round((1 - cosine), 2)

    def respond(self, text, intent_name):
        maximum = float('-inf')
        response = ""
        closest = ""
        replies = {}
        list_sim, list_replies = [], []
        dataset = df_chatbot[df_chatbot['Intent'] == intent_name]
        for i in dataset.iterrows():
            sim = self.cosine_distance_countvectorizer_method(text, i[1]['User'])
            list_sim.append(sim)
            list_replies.append(i[1]['Chatbot'])

        for i in range(len(list_sim)):
            if list_sim[i] in replies:
                replies[list_sim[i]].append(list_replies[i])
            else:
                replies[list_sim[i]] = list()
                replies[list_sim[i]].append(list_replies[i])
        d1 = sorted(replies.items(), key=lambda pair: pair[0], reverse=True)
        return d1[0][1][random.randint(0, len(d1[0][1]) - 1)]

    def extract_best_intent(self, list_intent_pred):
        intent_scores = {}
        for intent in self.Intents:
            intent_scores[intent] = 0.0
        for i in range(len(list_intent_pred)):
            intent_scores[list_intent_pred[i]] += self.model_weight[i]
        si = sorted(intent_scores.items(), key=lambda pair: pair[1], reverse=True)[:6]
        return si[0][0], round(si[0][1], 2)

    def get_human_names(self, text):
        person_list = []
        person_names = person_list
        tokens = nltk.tokenize.word_tokenize(text)
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary=False)

        person = []
        name = ""
        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
            for leaf in subtree.leaves():
                person.append(leaf[0])
            if len(person) > 0:  # avoid grabbing lone surnames
                for part in person:
                    name += part + ' '
                if name[:-1] not in person_list:
                    person_list.append(name[:-1])
                name = ''
            person = []
        # print (person_list)
        return person_list

    def replace_tag(self, text):
        text = text.replace('<HUMAN>', self.Human_name)

        # get current time
        BDT = pendulum.timezone('Asia/Dhaka')
        cdt = datetime.timetuple(datetime.now(BDT))
        hrs = int(cdt[3])
        am_pm = 'am'
        if int(cdt[3]) > 12:
            hrs = int(cdt[3]) - 12
            am_pm = 'pm'

        current_time = str(cdt[2]) + '-' + str(cdt[1]) + '-' + str(cdt[0]) + ' ' + str(hrs) + ':' + str(
            cdt[4]) + ' ' + am_pm
        text = text.replace('<TIME>', current_time)
        return text

    def chatbot_reply(self, text):
        processed_text = fe_cb.get_processed_text(text)

        if self.get_human_names(text):
            self.Human_name = self.get_human_names(text)[0]

        print('Intent using SVM: ', end='')
        svm_intent = svm_cb.predict(processed_text)[0]
        lr_intent = logisticRegr_cb.predict(processed_text)[0]
        dt_intent = dt_cb.predict(processed_text)[0]
        mnb_intent = mnb_cb.predict(processed_text)[0]
        xgbc_intent = xgbc_cb.predict(processed_text)[0]
        rfc_intent = rfc_cb.predict(processed_text)[0]
        mlp_intent = mlp_cb.predict(processed_text)[0]
        print(svm_intent)

        print('Intent using Logistic Regression: ', end='')
        print(lr_intent)
        print('Intent using Decision Tree: ', end='')
        print(dt_intent)
        print('Intent using Naive Bayes: ', end='')
        print(mnb_intent)
        print('Intent using XGBoost: ', end='')
        print(xgbc_intent)
        print('Intent using Random Forest: ', end='')
        print(rfc_intent)
        print('Intent using Multi-Layer Perceptron: ', end='')
        print(mlp_intent)

        # generating reply
        list_intent = [svm_intent, lr_intent, rfc_intent, xgbc_intent, mnb_intent, dt_intent, mlp_intent]
        best_intent, prob = self.extract_best_intent(list_intent)
        print('Best Intent:', best_intent, ':', prob)

        reply = "I don't understand. Please be specific" if prob < 0.4 else self.response_generate(text, best_intent)

        reply = self.replace_tag(reply)
        print('EDAIC:', reply)
        print()
        return reply, prob, best_intent

class Emotion:
    def __init__(self):
        self.Emotions = df_emotion['sentiment'].unique()
        accuracies = np.array([svm_summary_ed['Accuracy'], lr_summary_ed['Accuracy'], rfc_summary_ed['Accuracy'], xgbc_summary_ed['Accuracy'],
             mnb_summary_ed['Accuracy'], dt_summary_ed['Accuracy'], mlp_summary_ed['Accuracy']])
        norm_accuracy = accuracies - min(accuracies)
        self.emotion_model_weight = norm_accuracy/sum(norm_accuracy)

    def extract_best_emotion(self, list_emotion_pred):
        emotion_scores = {}
        for emotions in self.Emotions:
            emotion_scores[emotions] = 0.0
        for i in range(len(list_emotion_pred)):
            emotion_scores[list_emotion_pred[i]] += self.emotion_model_weight[i]
        se = sorted(emotion_scores.items(), key = lambda pair:pair[1],reverse=True)
        return se[0][0], round(se[0][1],2)

    def detect_emotion(self, text):
        processed_text = fe_ed.get_processed_text(text)

        svm_emotion = svm_ed.predict(processed_text)[0]
        lr_emotion = logisticRegr_ed.predict(processed_text)[0]
        dt_emotion = dt_ed.predict(processed_text)[0]
        mnb_emotion = mnb_ed.predict(processed_text)[0]
        xgbc_emotion = xgbc_ed.predict(processed_text)[0]
        rfc_emotion = rfc_ed.predict(processed_text)[0]
        mlp_emotion = mlp_ed.predict(processed_text)[0]

        list_emotion_pred = [svm_emotion, lr_emotion, rfc_emotion, xgbc_emotion, mnb_emotion, dt_emotion, mlp_emotion]
        best_emotion, prob = self.extract_best_emotion(list_emotion_pred)
        print('Best Emotion:',best_emotion,':',prob)

        print('Emotion using SVM: ',end = '')
        print(svm_emotion)
        print('Emotion using Logistic Regression: ',end = '')
        print(lr_emotion)
        print('Emotion using Decision Tree: ',end = '')
        print(dt_emotion)
        print('Emotion using Naive Bayes: ',end = '')
        print(mnb_emotion)
        print('Emotion using XGBoost: ',end = '')
        print(xgbc_emotion)
        print('Emotion using Random Forest: ',end = '')
        print(rfc_emotion)
        print('Emotion using Multi-Layer Perceptron ',end = '')
        print(mlp_emotion)
        print()
        return best_emotion, prob

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self.rate1, self.rate2, self.rate3, self.rate4, self.rate5 = None, None, None, None, None
        self.emo1, self.emo2, self.emo3, self.emo4, self.emo5, self.emo6, self.emo7 = None, None, None, None, None, None, None
        self._setup_main_window()
        self.chatbot = Chatbot()
        self.emotion = Emotion()
        self.bot_name = "EDAIC"
        self.msg_count = 0
        self.user_msg = ''              # user message
        self.reply = ''                 # chatbot reply
        self.best_intent = ''           # best intent after applying ensemble model
        self.chat_emotion = ''          # best emotion after applying ensemble model
        self.intent_prob = 0.0          # intent probability predicted by chatbot model
        self.emo_prob = 0.0             # emotion probability predicted by emotion detection model
        self.is_active_vote = False             # checks whether the getting vote is activated or not
        self.selected_rating = -1    #used to get selected rating from radio button as feedback
        self.selected_emotion = ''  #used to get selected emotion from radio button as feedback


    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit? All chats will be deleted"):
            self.store_feedback()
            self.window.destroy()
    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=700, height=730, bg=BG_COLOR)
        
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR_WHITE,
                           text="Emotionally Aware Chatbot - EDAIC", font=(FONT_BOLD, 16), pady=12)
        head_label.place(relwidth=1)
        
        # tiny divider
        line = Label(self.window, width=450, bg="#002366")
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg="#8CD3FF", fg=TEXT_COLOR_BLACK,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.6, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg="#FFC0CB", height=2)
        bottom_label.place(relwidth=1, rely=0.68, relheight=0.35)

        
        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#FFFFFF", fg=TEXT_COLOR_BLACK, font=FONT)
        self.msg_entry.place(relwidth=0.70, relheight=0.15, rely=0.008, relx=0.01)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = Button(bottom_label, text="Send", font=TEXT_COLOR_WHITE, width=15, fg=TEXT_COLOR_WHITE, bg=BUTTON_COLOR,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.715, rely=0.008, relheight=0.15, relwidth=0.27)

        # Emotion Label
        self.emotion = Label(bottom_label, bg=BG_COLOR, justify = tk.LEFT, fg=TEXT_COLOR_WHITE, font=FONT_BOLD, text="The emotion is: ")
        self.emotion.place(relwidth=0.30, relheight=0.15, rely=0.18, relx=0.01)

        # emotion widget label
        self.emotion_widget = Text(bottom_label, width=30, height=2, bg=BG_COLOR, fg=TEXT_COLOR_WHITE,
                                   font="Helvetica 15 bold italic", padx=190, pady=5)
        self.emotion_widget.place(relheight=0.15, relwidth=0.67, rely=0.18, relx=0.315)
        self.emotion_widget.configure(cursor="arrow", state=DISABLED)


        #Emotions Ratings

        def ShowFeedbackRating():
            print('Selected Rating:',rv.get())
            self.selected_rating = rv.get()

        def ShowFeedbackEmotion():
            print('Selected Emotion:',ev.get())
            self.selected_emotion = ev.get()


        self.rating_label = Label(bottom_label, bg=BG_COLOR, justify=tk.LEFT, fg=TEXT_COLOR_WHITE, font=FONT_BOLD,
                             text="Rate the emotion: ")
        self.rating_label.place(relwidth=0.298, relheight=0.15, rely=0.35, relx=0.01)


        ev = tk.StringVar() # emotion variable
        rv = tk.IntVar()
        # ★
        self.rate1 = Radiobutton(bottom_label,
                       text="1",
                       indicatoron=0,
                        bg=BUTTON_COLOR,
                        font=FONT_BOLD,
                        fg=TEXT_COLOR_WHITE,
                        variable =rv,
                        command=ShowFeedbackRating,
                        selectcolor=RADIO_BUTTON_COLOR,
                       value=1)
        self.rate1.place(relx=0.312,rely=0.35,relwidth=0.135, relheight=0.15)

        self.rate2 = Radiobutton(bottom_label,
                            text="2",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            highlightbackground="#FF0000",
                            selectcolor=RADIO_BUTTON_COLOR,
                            variable=rv,
                            command=ShowFeedbackRating,
                            value=2)
        self.rate2.place(relx=0.447, rely=0.35, relwidth=0.135, relheight=0.15)

        self.rate3 = Radiobutton(bottom_label,
                            text="3",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            variable=rv,
                            command=ShowFeedbackRating,
                            value=3)
        self.rate3.place(relx=0.582, rely=0.35, relwidth=0.135, relheight=0.15)

        self.rate4 = Radiobutton(bottom_label,
                            text="4",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            variable=rv,
                            command=ShowFeedbackRating,
                            value=4)
        self.rate4.place(relx=0.717, rely=0.35, relwidth=0.135, relheight=0.15)

        self.rate5 = Radiobutton(bottom_label,
                            text="5",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            variable=rv,
                            command=ShowFeedbackRating,
                            value=5)
        self.rate5.place(relx=0.852, rely=0.35, relwidth=0.135, relheight=0.15)

        # Emotion feedback
        line = Label(bottom_label, fg=TEXT_COLOR_WHITE, text="If the predicted emotion is not correct, choose the correct emotion:", font=FONT, bg="#002366")
        line.place(relwidth=1, rely=0.53, relheight=0.15)

        self.emo1 = Radiobutton(bottom_label,
                            text="Joy",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            command=ShowFeedbackEmotion,
                            variable=ev,
                            value="Joy")
        self.emo1.place(relx=0.01, rely=0.7, relwidth=0.14, relheight=0.15)

        self.emo2 = Radiobutton(bottom_label,
                            text="Love",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            highlightbackground="#FF0000",
                            selectcolor=RADIO_BUTTON_COLOR,
                            command=ShowFeedbackEmotion,
                            variable=ev,
                            value="Love")
        self.emo2.place(relx=0.15, rely=0.7, relwidth=0.14, relheight=0.15)

        self.emo3 = Radiobutton(bottom_label,
                            text="Surprise",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            command=ShowFeedbackEmotion,
                            variable=ev,
                            value="Surprise")
        self.emo3.place(relx=0.29, rely=0.7, relwidth=0.14, relheight=0.15)

        self.emo4 = Radiobutton(bottom_label,
                            text="Neutral",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            command=ShowFeedbackEmotion,
                            variable=ev,
                            value="Neutral")
        self.emo4.place(relx=0.43, rely=0.7, relwidth=0.14, relheight=0.15)

        self.emo5 = Radiobutton(bottom_label,
                            text="Sadness",
                            indicatoron=0,
                            bg=BUTTON_COLOR,
                            font=FONT_BOLD,
                            fg=TEXT_COLOR_WHITE,
                            selectcolor=RADIO_BUTTON_COLOR,
                            command=ShowFeedbackEmotion,
                            variable=ev,
                            value="Sadness")
        self.emo5.place(relx=0.57, rely=0.7, relwidth=0.14, relheight=0.15)

        self.emo6 = Radiobutton(bottom_label,
                           text="Fear",
                           indicatoron=0,
                           bg=BUTTON_COLOR,
                           font=FONT_BOLD,
                           fg=TEXT_COLOR_WHITE,
                           selectcolor=RADIO_BUTTON_COLOR,
                           command=ShowFeedbackEmotion,
                           variable=ev,
                           value="Fear")
        self.emo6.place(relx=0.71, rely=0.7, relwidth=0.14, relheight=0.15)

        self.emo7 = Radiobutton(bottom_label,
                           text="Anger",
                           indicatoron=0,
                           bg=BUTTON_COLOR,
                           font=FONT_BOLD,
                           fg=TEXT_COLOR_WHITE,
                           selectcolor=RADIO_BUTTON_COLOR,
                           command=ShowFeedbackEmotion,
                           variable=ev,
                           value="Anger")
        self.emo7.place(relx=0.85, rely=0.7, relwidth=0.14, relheight=0.15)

    def save_data(self, row_data):
        with open(desktop_path + 'Chat Feedback.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)

    def get_current_time(self):
        # get current time
        BDT = pendulum.timezone('Asia/Dhaka')
        cdt = datetime.timetuple(datetime.now(BDT))
        hrs = int(cdt[3])
        am_pm = 'am'
        if int(cdt[3]) > 12:
            hrs = int(cdt[3]) - 12
            am_pm = 'pm'

        current_time = str(cdt[2]) + '-' + str(cdt[1]) + '-' + str(cdt[0]) + ' ' + str(hrs) + ':' + str(
            cdt[4]) + ' ' + am_pm
        return current_time


    def store_feedback(self):
        if self.is_active_vote and (self.selected_rating != -1 or self.selected_emotion != ''):
            r = self.selected_rating if self.selected_rating != -1 else ''
            e = self.selected_emotion if self.selected_emotion != '' else ''
            '''
            row structure:
            datetime -- user_msg -- chatbot_reply -- predicted_intent -- predicted_emotion -- intent_prob -- emotion_prob
             -- rating -- feedback_emotion
            '''
            row = [self.get_current_time(), self.user_msg, self.reply, self.best_intent, self.chat_emotion,
                   self.intent_prob, self.emo_prob, r, e]
            self.save_data(row)
            self.selected_rating = -1
            self.selected_emotion = ''
            self.rate1.deselect()
            self.rate2.deselect()
            self.rate3.deselect()
            self.rate4.deselect()
            self.rate5.deselect()
            self.emo1.deselect()
            self.emo2.deselect()
            self.emo3.deselect()
            self.emo4.deselect()
            self.emo5.deselect()
            self.emo6.deselect()
            self.emo7.deselect()


    def _on_enter_pressed(self, event):
        self.store_feedback()
        self.user_msg = self.msg_entry.get()
        self.msg_count += 1
        self._insert_message(self.user_msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        self.reply, self.intent_prob, self.best_intent = self.chatbot.chatbot_reply(msg)
        self.chat_emotion, self.emo_prob = self.emotion.detect_emotion(msg)

        msg2 = f"{self.bot_name}: {self.reply}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.emotion_widget.configure(state=NORMAL)
        self.emotion_widget.delete('1.0', END)
        self.emotion_widget.insert(END, self.chat_emotion.strip().capitalize())
        self.emotion_widget.configure(state=DISABLED)
        self.is_active_vote = True

        self.text_widget.see(END)
        
if __name__ == "__main__":

    # Dataset
    print('Launching Chatbot...')
    df_chatbot = pd.read_csv(dataset_folder + 'Chatbot Dataset.csv', encoding='ISO-8859-1')
    df_chatbot = df_chatbot.dropna(axis=0)

    df_emotion = pd.read_csv(dataset_folder + 'text_emotions_neutral.csv')

    print('Chatbot Dataset length:',len(df_chatbot))
    print('Emotion Detection Dataset length:', len(df_emotion))
    # Train Test Split
    X_train_ed, X_test_ed, y_train_ed, y_test_ed = train_test_split(df_emotion['content'], df_emotion['sentiment'],
                                                                    test_size=0.3, random_state=116)
    X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(df_chatbot['User'], df_chatbot['Intent'],
                                                                    test_size=0.25, random_state=32)

    fe_cb = FeatureExtraction(rmv_stopword=False)
    fe_ed = FeatureExtraction(rmv_stopword=True)

    x_train_ed, x_test_ed = fe_ed.get_features(X_train_ed, X_test_ed)
    x_train_cb, x_test_cb = fe_cb.get_features(X_train_cb, X_test_cb)

    # Load Models
    chatbot_models = Models(x_train_cb, y_train_cb, x_test_cb, y_test_cb, model_name='cb')
    emotion_models = Models(x_train_ed, y_train_ed, x_test_ed, y_test_ed, model_name='ed')

    svm_cb, logisticRegr_cb, rfc_cb, xgbc_cb, mnb_cb, dt_cb, mlp_cb = chatbot_models.load_models()
    svm_summary_cb, lr_summary_cb, rfc_summary_cb, xgbc_summary_cb, mnb_summary_cb, dt_summary_cb, mlp_summary_cb = chatbot_models.model_summary()

    svm_ed, logisticRegr_ed, rfc_ed, xgbc_ed, mnb_ed, dt_ed, mlp_ed = emotion_models.load_models()
    svm_summary_ed, lr_summary_ed, rfc_summary_ed, xgbc_summary_ed, mnb_summary_ed, dt_summary_ed, mlp_summary_ed = emotion_models.model_summary()

    # Train Models and save into the disk
    # svm_cb, logisticRegr_cb, rfc_cb, xgbc_cb, mnb_cb, dt_cb, mlp_cb = chatbot_models.train_models()
    # svm_summary_cb, lr_summary_cb, rfc_summary_cb, xgbc_summary_cb, mnb_summary_cb, dt_summary_cb, mlp_summary_cb = chatbot_models.model_summary()
    # chatbot_models.save_models()
    #
    # svm_ed, logisticRegr_ed, rfc_ed, xgbc_ed, mnb_ed, dt_ed, mlp_ed = emotion_models.train_models()
    # svm_summary_ed, lr_summary_ed, rfc_summary_ed, xgbc_summary_ed, mnb_summary_ed, dt_summary_ed, mlp_summary_ed = emotion_models.model_summary()
    # emotion_models.save_models()

    print('\nChatbot \nSVM:',svm_summary_cb['Accuracy'], 'LR:',lr_summary_cb['Accuracy'], 'RFC:', rfc_summary_cb['Accuracy'],
          'XGBC:',xgbc_summary_cb['Accuracy'], 'NB:',mnb_summary_cb['Accuracy'],'DT:',dt_summary_cb['Accuracy']
          ,'MLP:',mlp_summary_cb['Accuracy'])
    print('Emotion Detection\nSVM:', svm_summary_ed['Accuracy'], 'LR:', lr_summary_ed['Accuracy'], 'RFC:',
          rfc_summary_ed['Accuracy'],
          'XGBC:', xgbc_summary_ed['Accuracy'], 'NB:', mnb_summary_ed['Accuracy'], 'DT:', dt_summary_ed['Accuracy']
          , 'MLP:', mlp_summary_ed['Accuracy'])
    app = ChatApplication()
    app.run()
