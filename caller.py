#Caller function to run
import os,sys
import argparse
from ig_api import *
from ai_model import chatbot
from keras.models import load_model
import pickle
import time
from cryptography.fernet import Fernet
sys.setrecursionlimit(20000)
import random
import string
from Classifier import *

def train_chatbot():
    bot = chatbot()
    bot.train("/Users/edwgent/Documents/Eddy_Backup/Documents/Hackathon/FutureOfHealth/firebase_key.json")

def train_classifier():
    classifier = Classifier('./opioid_data_2022.csv', './opioid_model_%s.pkl' %(datetime.datetime.now().strftime('%Y%m%d')),generate=True)

def assess_consent_sent(chats,user):
    for chat in chats:
        if chat['username'] == user:
            if chat['declaration'] == False:
                return True
    return False

def start_work():
    #chatbot model and dependencies...
    try:
        patient_bot = load_model('chatbot.patient.%s' %(datetime.datetime.now().strftime('%Y%m%d')))
        patient_classes = pickle.load(open('./chatbot_patient_classes.pkl','rb'))
        non_patient_bot = load_model('chatbot.non-patient.%s' %(datetime.datetime.now().strftime('%Y%m%d')))
        non_patient_classes = pickle.load(open('./chatbot_non-patient_classes.pkl','rb'))
        patient_words = pickle.load(open('./chatbot_patient_words.pkl', 'rb'))
        non_patient_words = pickle.load(open('./chatbot_non-patient_words.pkl', 'rb'))
    except IOError as e:
        logger.debug('Couldnt load model, retraining and regenerating...')
        logger.debug(e)
        train_chatbot()
        start_work()

    except Exception as ex:
        logger.debug('Couldnt load one or more models/dependencies - trying to retrain and try again...')
        logger.debug(ex)
        train_chatbot()
        start_work()

    #opioid prediction model and dependencies...
    try:
        opioid_model = pickle.load(open('opioid_model_%s.pkl' %(datetime.datetime.now().strftime('%Y%m%d')),'rb'))
    except IOError as e:
        logger.debug('Couldnt load model, retraining and regenerating...')
        logger.debug(e)
        train_classifier()
        start_work()



    # Pull intents from firebase, for both models
    cred = credentials.Certificate("/Users/edwgent/Documents/Eddy_Backup/Documents/Hackathon/FutureOfHealth/firebase_key.json")
    firebase_admin.initialize_app(cred,{'storageBucket': 'hack-4-rare.appspot.com'})

    db = firestore.client()
    intents_ref = db.collection(u'intents')
    docs = intents_ref.stream()

    intent_objects = []
    intents = []
    for intent in docs:
        intent_objects.append(intent.to_dict())

    for intent in intent_objects:
        intents.append(intent['intents'])

    bot = instagram_bot('exquisicarehealth',os.getenv('IG_PASSWORD'),2)
    bot.login()
    #bot.persist_to_firebase("/Users/edwgent/Documents/Eddy_Backup/Documents/Hackathon/FutureOfHealth/firebase_key.json",'analysis','querier')
    c_bot = chatbot()
    chats_ref = db.collection(u'chats')
    docs = chats_ref.stream()
    chats = [chat.to_dict() for chat in docs]

    def iterate():
        while 1:
            msgs = bot.gather_outstanding_messages()
            logger.info(msgs)

            for thread in msgs['inbox']['threads']:
                for msg in thread['items']:
                    if msg['is_sent_by_viewer'] == False:
                        inp = msg['text']
                        if inp.lower() == '' or inp.lower() == '*':
                            print('Please re-phrase your query!')
                            print("-" * 50)
                        else:
                            #Current issue - cant determine if all questions answered, takes first answer, finds its intent and then returns a likelihood/response based on it.
                            user = thread['users'][0]['username']

                            all_users = [chat['username'] for chat in chats]
                            if user in all_users:
                                pass
                            else:
                                chat_obj = {'username': user, 'declaration': False}
                                id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(20))
                                db.collection(u'chats').document(u'%s' % (id)).set(chat_obj)
                            logger.info('Message found - ', inp)
                            inp = str(inp).lower()
                            rsp=c_bot.chatbot_response(inp,patient_bot, non_patient_bot,patient_words,non_patient_words,patient_classes,non_patient_classes,intents)
                            print(f"Bot: {rsp}" + '\n')
                            print("-" * 50)
                            logger.info(user)
                            if type(rsp) == list: #Must be Approval intent
                                sent = assess_consent_sent(chats,user)
                                if sent:
                                    for response in rsp:
                                        if 'Thank' in response:
                                            first_to_send = response
                                    bot.send_message(first_to_send,user)
                                    for response in rsp:
                                        if response == first_to_send:
                                            continue
                                        else:
                                            bot.send_message(response,user)

                                else:
                                    message = """
                                    Just to let you know, we are not qualified medical professionals and are not qualified to offer official medical advice or recommendations.
However, the data and analysis we present to you are curated from multiple certified research institutes and Government bodies including the CDC.

We will never share any of your information to any other private or public third parties, but if you do not consent to use collecting your data, please respond 'No' to this message and this conversation will be terminated.
                                    """

                                    for chat in chats:
                                        if chat['username'] == user:
                                            chat['declaration'] = True
                                            db.collection(u'chats').document(u'%s' % (chat['id'])).set(chat)
                                    bot.send_message(message,user)
                            else:
                                bot.send_message(rsp,user)
                time.sleep(2)
                iterate()

    iterate()


def main():
    start_work()

if __name__ == '__main__':
    main()