from google.colab import files
import pandas as pd
import os

def load_kaggle_data(comp_data_api):
    files.upload()

    # Next, install the Kaggle API client.
    os.system('pip install -q kaggle')
    # The Kaggle API client expects this file to be in ~/.kaggle,
    # so move it there.
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')

    # This permissions change avoids a warning on Kaggle tool startup.
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    # Copy the desired data set locally. Go to the competition's paga, Data tab and copy the API
    # !kaggle competitions do
    os.system(comp_data_api)
    

def submit_prediction(comp_submission_api):
    files.upload()

    # Next, install the Kaggle API client.
    os.system('pip install -q kaggle')
    # The Kaggle API client expects this file to be in ~/.kaggle,
    # so move it there.
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')

    # This permissions change avoids a warning on Kaggle tool startup.
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system(comp_submission_api)
