import sys, os
import pickle
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from cryptography.fernet import Fernet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from firebase_admin import storage
class Classifier:

    def __init__(self, model_file,persist_loc,generate=False):
        self.model_file = model_file
        self.dependant = 'Opioid_Prscrbng_Rate'
        self.feature = 'Prscrbr_Geo_Desc'
        self.persist_loc = persist_loc
        if generate:
            self.generate_model()
        else:
            self.load_model()


    def predict_proba(self,input):
        count_vect = CountVectorizer()
        return self.model.predict(count_vect.transform([input]))
        
    @staticmethod
    def _Nmaxelements(list1, N):
        final_list = []

        for i in range(0, N):
            max1 = 0

            for j in range(len(list1)):
                if list1[j] > max1:
                    max1 = list1[j]

            list1.remove(max1)
            final_list.append(max1)

        return final_list

    def persist_model(self):
        print(self.persist_loc)
        with open(self.persist_loc, 'wb') as fh:
            pickle.dump(self.model, fh)


    def load_model(self):
        with open(self.persist_loc,'rb') as fh:
            self.model = pickle.load(fh)


    def generate_model(self):
        '''
        self.model_file_decrypted = self.model_file.replace('.csv','_decrypted.csv')
        try:
            key = os.getenv('MODEL_KEY')

        except Exception:
            print('MODEL_KEY env var not specified - cant decrypt output data and so cant train or generate model. Exiting')
            sys.exit()
        self.decrypt_data(self.model_file,key, self.model_file_decrypted)
        '''
        try:
            docs = pd.read_csv('./' + self.model_file)
        except Exception as ex:
            try:
               bucket = storage.bucket()
               blob = bucket.blob(self.model_file)
               blob.download_to_filename('./' + self.model_file)
               '''
               self.decrypt_data('./' + self.model_file,key, self.model_file_decrypted)
               '''
               docs = pd.read_csv('./' + self.model_file)
            except Exception as ex:
                print('Fatal error - cant find any model training data at %s path specified. Exiting!!' %(self.model_file))
                print(ex)
                sys.exit()
        docs = self.clean_dataset(docs)
        self.feature_set = docs[self.feature]
        print(self.feature_set.shape)
        self.dependants = docs[self.dependant]
        print(self.dependants.shape)
        mapping = {}
        self.user_mapping = mapping
        all_as_text = []
        for entry in list(self.user_mapping.values()):
            try:
                all_as_text.append(entry[0])
            except Exception:
                continue

        X_train, X_test, y_train, y_test = train_test_split(self.feature_set, self.dependants, random_state = 0)

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        self.model = MultinomialNB().fit(X_train_tfidf.astype(int), y_train.astype(int))
        self.persist_model()


    def clean_dataset(self,df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        return df



    def decrypt_data(self,path, key,  outpath = None):
        if outpath is None:
            outpath = '{}_encrypted'.format(path)
        f = Fernet(key)
        # opening the original file to encrypt
        with open(path, 'rb') as file:
            original = file.read()
        decrypted = f.decrypt(original)
        # opening the file in write mode and
        # writing the encrypted data
        with open(outpath, 'wb') as dfile:
            dfile.write(decrypted)
        print("file {} decrypted ; written at {} location".format(path, outpath))
