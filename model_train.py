import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from tensorflow import keras
import ujson
from lstm_model import model_generator
from lstm_model_v2_priceVolume import model_generator_v2_PV
from lstm_model_v2_priceVolumeVolatilityAdx import model_generator_v2_PVV
from fetchdata import fetch_data
import time as current_time
from dotenv import load_dotenv
from utils import get_model_save_path
from utils import ModelStorage
import pandas as pd

import pandas as pd
from utils import aggregate_5min_to_30min_custom
# Load environment variables from .env file
load_dotenv()

# Initialize storage
storage = ModelStorage()


def get_model_path(version: str, currency: str, model_id: str) -> str:
    # Always use the directory of this script as the base
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    version = version.strip().lower()
    currency = currency.strip().upper()
    model_id = model_id.strip()
    model_dir = os.path.join(MODELS_DIR, version, currency)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{model_id}.keras")

def handler(event, context):

    try:
        print("\n")
        print("-----------------------------------------")
        print("event ->")
        print(event)
        print("-----------------------------------------")
        print("\n")

        payload = ujson.loads(ujson.dumps(event))
        time = payload["time"]
        duration = payload["duration"]
        currency = payload["currency"]
        version = payload["version"]

        Window_size = duration * 2


        # Load your 5-min data (from Excel)
        df_5min = pd.read_excel(os.path.join(os.path.dirname(__file__), "btc_5min_data.xlsx"))

        

        if version in ["v1", "v2"]:
            # Aggregate to 30-min with offset 15 for aligned windows
            # processing on df_5min

            # convert end time to nearest 30 min
            end_time = (time//1800)*1800
            df_5min_endtime = end_time
            df_5min_starttime = end_time - (3*24*60*60) - (4*60*60) 
            
            print("Start time: ", df_5min_starttime)
            print("End time: ", df_5min_endtime)

            # Filter the DataFrame based on the start and end times
            df_5min = df_5min[(df_5min['TIMESTAMP'] >= df_5min_starttime) & (df_5min['TIMESTAMP'] <= df_5min_endtime)]
            print("Filtered DataFrame:")
            print(df_5min)
            print("\n")
            # Aggregate to 30-min with offset 0 for aligned windows
            data = aggregate_5min_to_30min_custom(df_5min)


        elif version in ["v1.1", "v2.1", "v2.2"]:
            # convert end time to nearest 5 min
            end_time = (time//300)*300
            df_5min_endtime = end_time
            df_5min_starttime = df_5min_endtime - (3*24*60*60) - (4*60*60)            
            print("Start time: ", df_5min_starttime)
            print("End time: ", df_5min_endtime)

            # Filter the DataFrame based on the start and end times
            df_5min = df_5min[(df_5min['TIMESTAMP'] >= df_5min_starttime) & (df_5min['TIMESTAMP'] <= df_5min_endtime)]
            print("Filtered DataFrame:")
            print(df_5min)
            print("\n")

             # Aggregate to 30-min with offset 0 for aligned windows
            data = aggregate_5min_to_30min_custom(df_5min)
            
           
        else:
            raise ValueError(f"Unknown version: {version}")

        # if version == "v1":
        #     data = fetch_data(time, currency, "v1")
        # elif version == "v1.1":
        #     data = fetch_data(time, currency, "v1.1")
        # elif version == "v2":
        #     data = fetch_data(time, currency, "v2")
        # elif version == "v2.1":
        #     data = fetch_data(time, currency, "v2.1")
        # elif version == "v2.2":
        #     data = fetch_data(time, currency, "v2.2")
        # else:
        #     raise ValueError(f"Unknown version: {version}")
        

        print("\n")
        print("-----------------------------------------")
        print("data ->")

        # # Convert data into a DataFrame with specified column names
        # data = pd.DataFrame(data, columns=['TIMESTAMP', 'CLOSE', 'VOLUME']) 
        # # Save the DataFrame to an Excel file
        # data.to_excel(
        #     rf"C:\Users\HarikrushnaSuhagiya\Downloads\BackTest-main\btc_training_data_{version}_{duration}.xlsx",
        #     index=False
        # )

        # Print the DataFrame
        print(data)
        print("-----------------------------------------")
        print("\n")
           
        # Set random seeds for reproducibility
        np.random.seed(0)
        tf.random.set_seed(0)
        random.seed(0)

        # Define the length of training data
        training_data_len = 144
        # print(data.shape)

        # Storing min and mx to scale
        volume_max_element = 1
        volume_min_element = 0
        volatility_max_element = 1
        volatility_min_element = 0
        adx_max_element = 1
        adx_min_element = 0

        if version in ["v1", "v1.1"]:
           price_max_element = np.max(data["CLOSE"])
           price_min_element = np.min(data["CLOSE"])

        elif version in ["v2", "v2.1","v2.2"]:
            price_max_element = np.max(data["CLOSE"])
            price_min_element = np.min(data["CLOSE"])
            volume_max_element = np.max(data["VOLUME"])
            volume_min_element = np.min(data["VOLUME"])
            

        if version in ["v1", "v1.1"]:
            # Only price
            if isinstance(data, pd.DataFrame):
                # If it's a DataFrame, extract CLOSE column
                close_data = data["CLOSE"].values
            else:
                # If it's a structured array, extract CLOSE field
                close_data = data["CLOSE"]
            
            # Reshape and scale only the CLOSE data
            close_data = close_data.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_data)
            
            # For v1/v1.1, we only need the scaled CLOSE data as a numpy array
            # The model_generator expects a 2D numpy array
            model = model_generator(version,scaled_data, training_data_len, Window_size, currency)

        elif version in ["v2", "v2.1"]:
            # Convert to DataFrame if it's a structured array
            if not isinstance(data, pd.DataFrame):
                training_data = pd.DataFrame({
                    'TIMESTAMP': data['TIMESTAMP'],
                    'CLOSE': data['CLOSE'],
                    'VOLUME': data['VOLUME']
                })
            else:
                training_data = data.copy()
            
            # For v2/v2.1, we need to prepare the data as expected by the model generators
            if Window_size < 8:
                # Convert to numpy array with both CLOSE and VOLUME
                model_data = training_data[['CLOSE', 'VOLUME']].values
                model = model_generator_v2_PV(model_data, training_data_len, Window_size, currency, version)
            else:
                # For larger window size, include volatility and ADX
                model_data = training_data[['CLOSE', 'VOLUME']].values
                model, volatility_max_element, volatility_min_element, adx_max_element, adx_min_element = model_generator_v2_PVV(
                    model_data, training_data_len, Window_size, currency, version
                )

        elif version == "v2.2":
            # Convert to DataFrame if it's a structured array
            if not isinstance(data, pd.DataFrame):
                training_data = pd.DataFrame({
                    'TIMESTAMP': data['TIMESTAMP'],
                    'CLOSE': data['CLOSE'],
                    'VOLUME': data['VOLUME']
                })
            else:
                training_data = data.copy()
            
            # Convert to numpy array with both CLOSE and VOLUME
            model_data = training_data[['CLOSE', 'VOLUME']].values
            model = model_generator_v2_PV(model_data, training_data_len, Window_size, currency, version)

        else:
            raise ValueError(f"Unknown version: {version}")

        
        print("Model Created")
        model.summary()

        # # Save model
        # try:
        #     model_path = get_model_path(version, currency, f"{currency}_{duration}")
        #     model.save(model_path)
        #     print(f"Model saved to: {model_path}")
            
        #     # # Upload to S3 if configured
        #     # bucket_name = os.getenv('RETRAIN_MODEL_S3_BUCKET')
        #     # if bucket_name:
        #     #     s3_key = f"{version}/{currency}/{currency}_{duration}.keras"
        #     #     s3_client = boto3.client('s3')
        #     #     s3_client.upload_file(model_path, bucket_name, s3_key)
        #     #     print(f"Model uploaded to S3: {bucket_name}/{s3_key}")
        # except Exception as save_error:
        #     print(f"Error saving model: {save_error}")
        #     raise save_error

        # model.save("version1 duration 24hrs.h5")

        # print(min_element)
        # print(max_element)
        # event["result"] = "RESULT"
        # event["result_data"] = ans.tolist()

        table_name = os.getenv('CRYPTO_TABLE')
        


        if version in ["v2", "v2.1", "v2.2"]:
            table_index = currency + "_" + str(version)+"_"+str(duration)
        else:
            table_index = currency + "_"+ str(version)+"_4"
            if duration <= 6:
                table_index = currency + "_" + str(version)+"_"+str(duration)

        current_time_str = str(current_time.time())
        
        print('Saving model metadata...')

        # Prepare metadata for local storage
        metadata = {
            'last_retrained_at': current_time_str,
            'max_element': float(price_max_element),
            'min_element': float(price_min_element),
        }
        
        if version in ["v2", "v2.1", "v2.2"]:
            metadata.update({
                'volume_max': float(volume_max_element),
                'volume_min': float(volume_min_element),
            })
        if version in ["v2", "v2.1"]:
            metadata.update({
                'volatility_max': float(volatility_max_element),
                'volatility_min': float(volatility_min_element),
                'adx_max': float(adx_max_element),
                'adx_min': float(adx_min_element)
            })
        from utils import get_model_save_path, get_metadata_save_path

        # Save model
        model_path = get_model_save_path(currency, duration, version)
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        # Save metadata
        metadata_path = get_metadata_save_path(f"{currency}_{version}_{duration}")
        storage.save_metadata(f"{currency}_{version}_{duration}", metadata)
        print(f"Metadata saved locally to: {metadata_path}")

        # # Save to local storage
        # storage.save_metadata(table_index, metadata)
        # print(f"Metadata saved locally to: {storage.get_metadata_save_path(f"{currency}_{version}_{duration}")}")
        # print("Model training and metadata storage completed successfully")
        
        return {
            'statusCode': 200,
            'body': f'Model and metadata saved successfully for {currency}_{duration}'
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

