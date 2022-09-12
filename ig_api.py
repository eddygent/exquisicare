#Instagram messaging bot
#Periodically sweewhich pip3s through outstanding messages and runs some analyses to best advise on addiction affinity towards opioids

import os, sys
from instabot import Bot
import firebase_admin
from firebase_admin import auth
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import logging
import traceback

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

'''
users_ref = db.collection(u'users')
docs = users_ref.stream()
'''


class instagram_bot:
    def __init__(self,account, password,retry_connect):
        self.username = account
        self.password = password
        self.bot = Bot()
        self.reconnect = retry_connect
        self.attempts = 0

    def login(self):
        try:
            self.bot.login(username=self.username, password=self.password,
                      proxy=None,use_cookie=False)
            logger.info('logged in!')
            return
        except Exception as ex:
            if self.attempts <= self.reconnect:
                logger.info('Unable to login to IG - %s. Will retry %s times total before dying' %(traceback.format_exc(),self.reconnect))
                self.attempts += 1
                self.login()
            else:
                logger.error('Cant login - exceeded max number of retries. Exiting.')
                sys.exit(self.reconnect)

    def gather_outstanding_messages(self):
        return self.bot.get_messages()

    def send_message(self,text, userId):
        return self.bot.send_message(text,userId)

    def persist_to_firebase(self,firebaseKey, collection, userType):

        cred = credentials.Certificate(firebaseKey)
        firebase_admin.initialize_app(cred)

        db = firestore.client()
        intents_ref = db.collection(u'intents')
        docs = intents_ref.stream()

        app_clients = []
        for user in docs:
            app_clients.append(user.to_dict())
