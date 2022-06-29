
from multiprocessing.connection import wait
import os
import csv
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def firebase_to_csv(data_directory, csv_destination_directory):
    data_files = os.listdir(data_directory)

    columns = ['ind','Activity','Accel. X','Accel. Y','Accel. Z','Gyro. X','Gyro. Y','Gyro. Z']

    if not os.path.exists(csv_destination_directory):
        os.makedirs(csv_destination_directory)
        
    f_csv = open(f'{csv_destination_directory}/data.csv','w')
    writer = csv.writer(f_csv)
    writer.writerow(columns)
    i=1

    for aux_f in data_files:
        f = open(f'{data_directory}/{aux_f}', 'r')
        lines = f.readlines()
        activity = lines[0].split()[0] #split é para remover o \n
        
        for line in lines[1::]:
            row = []
            row.append(i)
            i+=1
            row.append(activity)
            aux_row = line.split(',')
            for n in aux_row:
                row.append(n.split()[0])
            
            writer.writerow(row)
        f.close()

    f_csv.close()

def preprocess_csvdata(data_dataframe):
    

    ###############################################################################################################
    # Now that the csv file win un-processed data has been created, we can read it as a Pandas Dataframe and make some changes to the data
    FRAME_SIZE = 200

    # Remove the index column
    data_dataframe = data_dataframe.drop('ind', axis=1, inplace=False)


    # Split data into X/Input y/output/Target sets
    X = data_dataframe.drop('Activity', axis=1, inplace=False)
    y = pd.DataFrame(data_dataframe['Activity'])

    # Encode Target labels ('corrida', 'andar', etc...)
    y = pd.get_dummies(y)

    # Scale input values (the axis values)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = pd.DataFrame(scaler.fit_transform(X=X))
    

    '''
        By default, each frame has 1000 entries.
        We'll split each one in half, so that each frame/instance contains only 200 entries.
        This has to be done on both input and ouput data sets.
        Each frame will have 200 rows and 6 columns (6 sensor axis).
        Each frame will be associated with a vector of length 4 indicating its corresponding activity/label.
    '''
    print(f"Splitting X and y of shapes {X.shape} and {y.shape}")
    X, y = split_frames(X, y, frame_size=FRAME_SIZE, input_features=6, output_classes=4)

    return X, y

    
def split_frames(X, y, frame_size, input_features, output_classes):
    model_input = []
    model_output = []

    # Model input are (frame_size x input_features) matrices
    instances = X.shape[0]

    for index in range(0, instances, frame_size):

        # Get 'frame_size' entries of input data
        input_frame_data = X.iloc[index:index + frame_size].to_numpy(copy=True)
        model_input.append(input_frame_data.reshape(
            frame_size, input_features))

        # For each input frame, we have a label/activity to match.
        output_frame_label = y.iloc[[index]].to_numpy(copy=True)
        model_output.append(output_frame_label.reshape(output_classes))

    model_input = np.array(model_input)
    model_output = np.array(model_output)
    return model_input, model_output

def augment_running_data(dataframe: pd.DataFrame):

    original_dataframe = dataframe.copy(deep=True)

    # Get rows of activity 'running'
    running_rows = original_dataframe[original_dataframe['Activity'] == 'correr']

    # Apply symmetry transformation
    transformed_running_rows = running_rows
    transformed_running_rows['Accel. X'] = -transformed_running_rows['Accel. X'] 
    transformed_running_rows['Accel. Z'] = -transformed_running_rows['Accel. Z'] 
    transformed_running_rows['Gyro. X'] = -transformed_running_rows['Gyro. X'] 
    transformed_running_rows['Gyro. Z'] = -transformed_running_rows['Gyro. Z'] 

    # Insert these transformed rows ino the dataframe
    original_dataframe = pd.concat([original_dataframe, transformed_running_rows], ignore_index=True)
    return original_dataframe
