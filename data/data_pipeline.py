
import data_download
import data_preprocessing
import sys
import os



if __name__ == '__main__':

    options = sys.argv
    print(f'Options used: {options}')

    DOWNLOADED_DATA_DIR = "./downloaded_data"

    if 'download' in options:
        print("=> Downloading data files...")
        data_download.download_data(destination_directory=DOWNLOADED_DATA_DIR)

    if 'preproc' in options :
        print("=> Preprocessing data files...")
        data_preprocessing.firebase_to_csv(
            data_directory=DOWNLOADED_DATA_DIR, 
            csv_destination_directory=DOWNLOADED_DATA_DIR
        )

    if 'viz' in options:
        print("=> Starting data visualization...")
        os.system("streamlit run data_viz.py")

