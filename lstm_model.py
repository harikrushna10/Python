import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.regularizers import l2
from utils import get_model_save_path

def model_generator(version,scaled_data,training_data_len,Window_size,currency):

    duration = int(Window_size / 2)

    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(0)

    bucket_name = os.getenv('RETRAIN_MODEL_S3_BUCKET')  # Update with your bucket name
    model_key = f"{'v1'}/{currency}/{currency}_{duration}.keras"
    print(model_key)
    
   # train_data = scaled_data[:training_data_len, :] #confirn with jagdish sir
    train_data = scaled_data[-training_data_len:] #confirn with jagdish sir
    x_train = []
    y_train = []

    for i in range(Window_size, len(train_data) - Window_size):
        x_train.append(train_data[i - Window_size : i, 0])
        y_train.append(train_data[i : i + Window_size, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(Window_size))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(x_train, y_train, batch_size=1, epochs=5)

    model_path = get_model_save_path(currency, duration, version)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Comment out any S3 upload code if present
    # if bucket_name:
    #     s3_key = f"v1/{currency}/{currency}_{duration}.keras"
    #     s3_client.upload_file(model_path, bucket_name, s3_key)
    
    return model