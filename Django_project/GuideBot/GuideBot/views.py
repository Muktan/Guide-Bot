from django.shortcuts import render, render_to_response
from django.views.generic import TemplateView
from django.http import HttpResponseRedirect
from django.contrib import auth
from django.template.context_processors import *
from GuideBot.models import *
from django.core.mail import send_mail
from django.conf import settings
import smtplib
import math, random

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import urllib.request
import nltk
import speech_recognition as sr
from google_speech import Speech
import tensorflow as tf
with open("C:\\Users\\Deepang\\OneDrive\\Desktop\\Guide-Bot\\Django_project\\GuideBot\\GuideBot\\intents.json") as file:
    data = json.load(file)


datamain = []
c_data = []
u_data = []
def Welcome(request):
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)


        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load("model.tflearn")
    #model.fit(training,output,n_epoch=1000,batch_size=10,show_metric=True)
    #model.save("model.tflearn")
    

    
    def hello ():
        r = sr.Recognizer()                                                                                   
        with sr.Microphone() as source:                                
            print("Speak:")                                                                                   
            audio = r.listen(source)
            text = ""
        try:
            text = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        def bag_of_words(s, words):
            bag = [0 for _ in range(len(words))]
            s_words = nltk.word_tokenize(s)
            s_words = [stemmer.stem(word.lower()) for word in s_words]
            for se in s_words:
                for i, w in enumerate(words):
                    if w == se:
                        bag[i] = 1
            return numpy.array(bag)
        def chat():
            print("Start talking with the bot (type quit to stop)!")
            results = model.predict([bag_of_words(text, words)])
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            return random.choice(responses)
        c_text=str(chat())
        lang = "en"
        speech = Speech(c_text, lang)
        sox_effects = ("speed", "0.95")
        speech.play(sox_effects)
        return c_text,text

    c_text,u_text = hello()
    # datamain.append(u_text)
    # datamain.append(c_text)
    datamain.insert(0,c_text)
    datamain.insert(0,u_text)
    
    

    return render(request, "Welcome.html", {"data": datamain})