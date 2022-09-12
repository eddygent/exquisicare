Exquisicare chatbot - solving the opioid crisis through a DM.

This chatbot interacts with the Instagram library to respond to enquiries over opioid addiction stats and offer advice through a classification model trained on CDC data.

You'll need to install the requirements into your venv:

pip install requirements.txt

You'll also need to change the default path of the credentials json file for authenticating with our Firebase backend to use one of your own (this is where model intents, keywords and chat metadata is stored), or you can request the Exquisicare team to send you over the key.

You'll need to also request for the password to the Instagram account, and then set it to your Bash environment with:

export IG_PASSWORD=xxxxx

Finally, run the backend with:

python caller.py 
