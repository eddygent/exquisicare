import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.chat.util import Chat, reflections
import json
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import datetime
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.tokenize import sent_tokenize, word_tokenize
from Classifier import *


import random
import logging
import re

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class chatbot:
    def __init__(self):
        pass

    def parse_documents_and_classes(self,intents,modelType):
        words = []
        classes = []
        documents = []
        self.ignore_words = ['?','!']
        for intent in intents:
            for pattern in intent['patterns']:
                w= nltk.word_tokenize(pattern)
                words.extend(w)
                documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

        self.models[modelType]['classes'] = classes
        self.models[modelType]['documents'] = documents
        self.models[modelType]['words'] = words

    def create_training_data(self,lemmatizer, modelType):
        training_data = []
        empty_op_array = [0] * len(self.models[modelType]['classes'])
        for doc in self.models[modelType]['documents']:
            bow = []
            pattern_words = doc[0]
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in self.models[modelType]['words']:
                bow.append(1) if w in pattern_words else bow.append(0) #binary classification

            op_row = list(empty_op_array)
            op_row[self.models[modelType]['classes'].index(doc[1])] = 1
            training_data.append([bow, op_row])
        random.shuffle(training_data)
        training_data = np.array(training_data)

        self.models[modelType]['x_train'] = list(training_data[:,0])
        self.models[modelType]['y_train'] = list(training_data[:,1])
        logger.info('Created training data set for model %s of size %s' %(modelType, np.size(self.models[modelType]['x_train'])))

    def lemmatize(self,modelType):
        lemmatizer= WordNetLemmatizer()
        words  = self.models[modelType]['words']
        classes = self.models[modelType]['classes']
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in self.ignore_words]
        self.models[modelType]['words'] = sorted(list(set(words)))
        self.models[modelType]['classes'] = sorted(list(set(classes)))
        self.create_training_data(lemmatizer, modelType)

    def clean_up_sentence(self,sentence):
        lemmatizer= WordNetLemmatizer()

        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)

        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self,sentence, words, show_details=True):

        # tokenize the pattern
        #sentence_words = self.clean_up_sentence(sentence)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence:
            for i, w in enumerate(words):
                if w == s:

                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return (np.array(bag))


    def predict_class(self,sentence, model, words, classes):
        #Determines the user origin and selects model accordingly based on highest intent class confidence from both
        # filter out predictions below a threshold
        p = self.bow(sentence, words, show_details=False)
        logger.info(p)
        res = model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error]
        logger.info('Results', results)
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def create_neural_net(self,modelType):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.models[modelType]['x_train'][0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.models[modelType]['y_train'][0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        hist =model.fit(np.array(self.models[modelType]['x_train']), np.array(self.models[modelType]['y_train']),epochs=200,batch_size=5,verbose=1)
        model.save('chatbot.%s.%s' %(modelType,datetime.datetime.now().strftime('%Y%m%d')), hist)
        with open('./chatbot_%s_classes.pkl' %(modelType),'wb') as fh:
            pickle.dump(self.models[modelType]['classes'], fh)

        ##Upload to firebase storage

    def train(self,firebaseKey):
        #Pull intents from firebase, train booth patient/non-patient NNs
        cred = credentials.Certificate(firebaseKey)
        firebase_admin.initialize_app(cred)

        db = firestore.client()
        intents_ref = db.collection(u'intents')
        docs = intents_ref.stream()

        intents = []
        self.models = {}
        for intent in docs:
            intents.append(intent.to_dict())

        for intent in intents:
            modelType = intent['type']
            self.models[modelType] = {}
            intents = intent['intents']
            self.parse_documents_and_classes(intents,modelType)
            self.lemmatize(modelType)
            self.create_neural_net(modelType)

            with open('./chatbot_%s_words.pkl' %(modelType), 'wb') as fh:
                pickle.dump(self.models[modelType]['words'], fh)
            logger.info('Completed %s training workflow' %(modelType))

    def getResponse(self,ints, intents_json,return_all=False):
        tag = ints[0]['intent']
        list_of_intents = intents_json
        for int_list in list_of_intents:
            for i in int_list:
                logger.info(i)
                if (i['tag'] == tag):
                    if return_all:
                        result = i['responses']
                    else:
                        result = random.choice(i['responses'])
                    break
        return result

    def chatbot_response(self,text, patient_model,non_patient_model, patient_words,non_patient_words, patient_classes,non_patient_classes,intents):
        text = self.clean_up_sentence(text)
        ints_patient = self.predict_class(text, patient_model,patient_words,patient_classes)
        ints_non_patient = self.predict_class(text, non_patient_model,non_patient_words, non_patient_classes)
        '''
        if float(ints_patient[0]['probability']) > float(ints_non_patient[0]['probability']):
            res = self.getResponse(ints_patient, intents)
        else:
        '''
        logger.info('Intent returned: ')
        logger.info( ints_non_patient)
        if ints_non_patient[0]['intent'] in ['Approval']:
            res = self.getResponse(ints_non_patient,intents,return_all=True)
        elif ints_non_patient[0]['intent'] == 'Age':
            self.age = re.match(r'([0-9]*)', text).group(0)
            classifier = Classifier('./opioid_data_2022.csv',
                                    './opioid_model_%s.pkl' % (datetime.datetime.now.strftime('%Y%m%d')))
            proba = classifier.predict_proba(self.age)
            res = "Based on our calculations, you have approximately a %s chance of opioid addiction. Fortunately, there are many things you can do to reduce this lieklihood, including working closely with experienced MDs within the field who can help you understand the factors influencing addiction and healthy, sustainable steps you can take to manage them. Would you like us to connect you with a Primary Care specialist?" % (
            proba[0])

        elif ints_non_patient[0]['intent'] == 'Region':
            self.region = re.match(r'([a-zA-Z]*)', text).group(0)
            classifier = Classifier('./opioid_data_2022.csv','./opioid_model_%s.pkl' %(datetime.datetime.now.strftime('%Y%m%d')))
            proba = classifier.predict_proba(self.region)
            res = "Based on our calculations, you have approximately a %s chance of opioid addiction. Fortunately, there are many things you can do to reduce this lieklihood, including working closely with experienced MDs within the field who can help you understand the factors influencing addiction and healthy, sustainable steps you can take to manage them. Would you like us to connect you with a Primary Care specialist?" %(proba[0])
        else:
            res = self.getResponse(ints_non_patient,intents)
        return res

