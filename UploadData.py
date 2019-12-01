from google.colab import files
import pandas as pd

def load_Kaggle2Collab(comp_api):
    files.upload()

    # Next, install the Kaggle API client.
    !pip install -q kaggle
    # The Kaggle API client expects this file to be in ~/.kaggle,
    # so move it there.
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/

    # This permissions change avoids a warning on Kaggle tool startup.
    !chmod 600 ~/.kaggle/kaggle.json
    # Copy the desired data set locally. Go to the competition's paga, Data tab and copy the API
    # !kaggle competitions download -c titanic
    !$comp_api