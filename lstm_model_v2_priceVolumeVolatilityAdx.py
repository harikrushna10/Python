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


def calculate_volatility(close_prices):
    """Calculate volatility based on the log returns of the closing prices."""
    if len(close_prices) < 2:
        return np.nan
    log_returns = np.diff(np.log(close_prices))
    volatility = np.std(log_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
    return volatility

def calculate_adx(close_prices):
    """Calculate the Average Directional Index (ADX) based on the closing prices."""
    if len(close_prices) < 2:
        return np.nan
    price_changes = np.diff(close_prices)
    if len(price_changes) < 3:
        return np.nan
    smooth_changes = pd.Series(price_changes).rolling(window=3).mean().dropna()
    if smooth_changes.empty:
        return np.nan
    plus_di = smooth_changes.where(smooth_changes > 0, 0)
    minus_di = smooth_changes.where(smooth_changes < 0, 0)
    directional_index = np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = directional_index.rolling(window=3).mean().iloc[-1] if len(directional_index) > 0 else np.nan
    return adx

def model_generator_v2_PVV(data,training_data_len,window_size,currency,version):

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

    volatility = []
    for i in range(8, len(df)):
        volatility.append(calculate_volatility(df['CLOSE'][i-8:i]))

    adx = []
    for i in range(8, len(df)):
        adx.append(calculate_adx(df['CLOSE'][i-8:i]))

    df['VOLATILITY'] = [np.nan]*8 + volatility
    df['ADX'] = [np.nan]*8 + adx
    df.dropna(inplace=True)
    

    volatility_max_element = np.max(df["VOLATILITY"])
    volatility_min_element = np.min(df["VOLATILITY"])
    adx_max_element = np.max(df["ADX"])
    adx_min_element = np.min(df["ADX"])
    
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    scaler_volatility = MinMaxScaler(feature_range=(0, 1))
    scaler_adx = MinMaxScaler(feature_range=(0, 1))
    

    df['price_scaled'] = scaler_price.fit_transform(df[['CLOSE']])
    df['volume_scaled'] = scaler_volume.fit_transform(df[['VOLUME']])
    df['volatility_scaled'] = scaler_volatility.fit_transform(df[['VOLATILITY']])
    df['adx_scaled'] = scaler_adx.fit_transform(df[['ADX']])

    # Calculate volatility (standard deviation of the last 8 prices)

    # Add volatility to the DataFrame (the first 8 values will be NaN)

    # Drop NaN values (because the first 8 rows won't have volatility)

    # Convert DataFrame to numpy array for LSTM input
    scaled_data = df[['price_scaled', 'volume_scaled', 'volatility_scaled','adx_scaled']].values
    
    x_train = []
    y_train = []


    for i in range(len(scaled_data) - 2 * window_size):
        x_train.append(scaled_data[i:i+window_size])
        y_train.append(scaled_data[i+window_size:i+(2*window_size), 0])  # Predict price (the next price)
    
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 4))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 4)))
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
    epoch  = 0
    while current_loss > 0.1 or ct<=5:
        ct+=1
        epoch += 1
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)    
        current_loss = history.history['loss'][0]  # Get the loss from the current epoch
        print(f"Epoch {epoch}: Loss = {current_loss}")
        if current_loss <= 0.1:
            print(f"Stopping training as loss has reached the threshold of {0.1}")
            
    print("total epoch run")
    print(epoch)
   
    model_path = get_model_save_path(currency, duration, version)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Comment out any S3 upload code if present
    # if bucket_name:
    #     s3_key = f"v2/{currency}/{currency}_{duration}.keras"
    #     s3_client.upload_file(model_path, bucket_name, s3_key)
    
    return model, volatility_max_element, volatility_min_element, adx_max_element, adx_min_element



# data = data_calculator(1721829600,"BTC-USD","v2")

# model = model_generator_v2_PVV(data,50,8,"BTC-USD","v2")

# print(model.summary())