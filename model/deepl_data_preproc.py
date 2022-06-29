import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler



def read_data_n_preproc(dataframe=None, data_csv_path=None, test_split=0.1, validation_split=0.3, validation_split_seed=42, verbose=False):

    '''
    This should be used for training/validation only, has it trains the model with fewer data.
    If you wish to deploy the model with the full available data, use `read_full_data_n_preproc`
    '''    
    data = None
    if data_csv_path is not None:
        # Read .csv file into mem.
        csv_data = pd.read_csv(data_csv_path)
    elif dataframe is not None:
        data = dataframe.copy(deep=True)

    # Remove index column
    data = csv_data.drop('ind', axis=1, inplace=False)

    #data = augment_running_data(data)

    # Split data into X/Input y/output/Target sets
    X = data.drop('Activity', axis=1, inplace=False)
    y = pd.DataFrame(data['Activity'])

    # Encode Target labels ('corrida', 'andar', etc...)
    y = pd.get_dummies(y)

    saved_y = y

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
    X, y = split_frames(X, y, frame_size=200, input_features=6, output_classes=4)

  

    # Now we need to define training, validation and testing sets
    STATE_SEED = 123
    X_train, X_test, y_train, y_test = train_test_split(
                                                            X, 
                                                            y, 
                                                            test_size=test_split, 
                                                            shuffle=True, 
                                                            random_state=STATE_SEED, 
                                                            stratify=y)

    X_train, X_validation, y_train, y_validation = train_test_split(
                                                                    X_train, 
                                                                    y_train, 
                                                                    test_size=validation_split, 
                                                                    shuffle=True, 
                                                                    random_state=validation_split_seed,
                                                                    stratify=y_train)

    # We're good to go!
    if verbose:
        print("============== Data Info. ==============")
        print(f"=> Read CSV with {csv_data.shape[0]} entries")
        print(f"=> Input/X data has shape  {X.shape}")
        print(f"=> Output/y data has shape  {y.shape}")
        print("----------------------------------------")
        print(f"=> Training Input has shape  {X_train.shape}")
        print(f"=> Training Output has shape  {y_train.shape}")
        print("----------------------------------------")
        print(f"=> Validation Input has shape  {X_validation.shape}")
        print(f"=> Validation Output has shape  {y_validation.shape}")
        print("----------------------------------------")
        print(f"=> Testing Input has shape  {X_test.shape}")
        print(f"=> Testing Output has shape  {y_test.shape}")
        print("=========================================")

    return X_train, y_train, X_validation, y_validation, X_test, y_test, saved_y

def read_full_data_n_preproc(dataframe=None, data_csv_path=None, verbose=False):
    data = None
    if data_csv_path is not None:
        # Read .csv file into mem.
        csv_data = pd.read_csv(data_csv_path)
    elif dataframe is not None:
        data = dataframe.copy(deep=True)

    # Remove index column
    data = csv_data.drop('ind', axis=1, inplace=False)

    #data = augment_running_data(data)

    # Split data into X/Input y/output/Target sets
    X = data.drop('Activity', axis=1, inplace=False)
    y = pd.DataFrame(data['Activity'])

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
    X, y = split_frames(X, y, frame_size=200, input_features=6, output_classes=4)

  

    # We're good to go!
    if verbose:
        print("============== Data Info. ==============")
        print(f"=> Read CSV with {csv_data.shape[0]} entries")
        print(f"=> Input/X data has shape  {X.shape}")
        print(f"=> Output/y data has shape  {y.shape}")
        print("=========================================")

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
    transformed_running_rows['Gyro. Y'] = -transformed_running_rows['Gyro. Y'] 
    transformed_running_rows['Gyro. Z'] = -transformed_running_rows['Gyro. Z'] 

    # Insert these transformed rows ino the dataframe
    original_dataframe = pd.concat([original_dataframe, transformed_running_rows], ignore_index=True)
    return original_dataframe
