import pyrebase
from os import listdir
from os.path import isfile, join
import os


def download_data(destination_directory):
  print("==> Downloading data files from Firebase. This might take a while... (don't interrupt!)")

  config = {
    "apiKey": "e3c1e32ade539d5eed84e1b642f1511bef99563f",
    "authDomain": "projectId.firebaseapp.com",
    "databaseURL": "https://DataCollector.firebaseio.com",
    "storageBucket": "datacollector-127c5.appspot.com",
    "serviceAccount": "../credentials/credentials.json"
  }

  firebase=pyrebase.initialize_app(config)
  storage = firebase.storage()

  # Create the directory for the downloaded data
  if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
  
  downloaded_files = [f for f in listdir(f'{destination_directory}/') if isfile(f'{destination_directory}/{f}')]

  all_files = storage.child("Data").list_files()
  for file in all_files:            
      try:
          name = str(file.name).split("/")[1].replace(":","_")
          if name not in downloaded_files:
                  storage.child(file.name).download(f'{destination_directory}/{name}')
          else:
              print(f"File {name} already downloaded. Skipping it")
      except:    
              print(f'{file.name} Download Failed')
  print("==> Data files downloaded.")
              