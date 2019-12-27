import tkinter as tk
root= tk.Tk()
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
nltk.download('punkt')
import speech_recognition as sr
from google_speech import Speech   
canvas1 = tk.Canvas(root, width = 300, height = 300)
canvas1.pack()
with open("intents.json") as file:
    data = json.load(file)

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
    text=str(chat())
    lang = "en"
    speech = Speech(text, lang)
    sox_effects = ("speed", "0.9")
    tokens = [t for t in text.split()]
    print(tokens)
    #freq = nltk.FreqDist(tokens)
    checker=""
    i=0
    for key in tokens:
        checker+=key+" "
        print(str(key))
        i+=1
        if(i%10==0):
            checker+="\n"
    frame = tk.Frame(root,width=1000,height=500)
    label1 = tk.Label(frame)
    label1['text']=" "
    button1 = tk.Button(frame,text='Ask Me',command=hello, bg='brown',fg='white')
    label1 = tk.Label(frame, text= checker, bg="yellow", fg='green', font=('helvetica', 12, 'bold'))
    button1.place(x=500,y=150,anchor="center")
    label1.place(x=500,y=250,anchor="center")
    canvas1.create_window(150, 200, window=frame)
    speech.play(sox_effects)
    
button1 = tk.Button(text='Click Me',command=hello, bg='brown',fg='white')
canvas1.create_window(150, 150, window=button1)
root.mainloop()
