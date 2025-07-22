import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout
from keras.regularizers import l2
from utils import get_model_save_path


def model_generator_v2_PV(data,training_data_len,window_size,currency,version):

    duration = int(window_size / 2)

    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)

    bucket_name = os.getenv('RETRAIN_MODEL_S3_BUCKET')  # Update with your bucket name
    model_key = f"{version}/{currency}/{currency}_{duration}.keras"
    print(model_key)
    

    # EXPECTED DATA FORMAT
    # data = np.array(
    #     [(entry["CLOSE"], entry["VOLUME"]) for entry in json_response["Data"]],
    #     dtype=[("CLOSE", "f4"), ("VOLUME", "f4")]  # Define named columns
    # )
   
    df = pd.DataFrame(data, columns=['CLOSE', 'VOLUME'])

    scaler_price = MinMaxScaler(feature_range=(0, 1))
    scaler_volume = MinMaxScaler(feature_range=(0, 1))

    df['price_scaled'] = scaler_price.fit_transform(df[['CLOSE']])
    df['volume_scaled'] = scaler_volume.fit_transform(df[['VOLUME']])

    scaled_data = df[['price_scaled', 'volume_scaled']].values

    scaled_data = scaled_data[-training_data_len:]
    
    x_train = []
    y_train = []

    

    for i in range(len(scaled_data) - 2 * window_size):
        x_train.append(scaled_data[i:i+window_size])
        y_train.append(scaled_data[i+window_size:i+(2*window_size), 0])  # Predict price (the next price)
    
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 2))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 2)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(window_size))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model until loss is greater than or equal to 0.1
    current_loss = float('inf')  # Set to a large value initially
    
    ct = 0
    epoch = 0
    while current_loss > 0.1 or ct<=5:
        ct+=1
        epoch += 1
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)    
        current_loss = history.history['loss'][0]  # Get the loss from the current epoch
        print(f"Epoch {epoch}: Loss = {current_loss}")
        if current_loss <= 0.1:
            print(f"Stopping training as loss has reached the threshold of {0.1}")

    print("total epochs run")        
    print(epoch)
    # New (centralized)
    from utils import get_model_save_path, get_metadata_save_path

    model_path = get_model_save_path(currency, duration, version)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Comment out any S3 upload code if present
    # if bucket_name:
    #     s3_key = f"v2/{currency}/{currency}_{duration}.keras"
    #     s3_client.upload_file(model_path, bucket_name, s3_key)
    return model


# data = data_calculator(1721829600,"BTC-USD","v2")

# model = model_generator_v2_PV(data,50,8,"BTC-USD","v2")

# print(model.summary())