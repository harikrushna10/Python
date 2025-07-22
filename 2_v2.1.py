


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import os
import time
import json
from dotenv import load_dotenv
from model_train import handler
from test_model import handler_test
from fetchdata import fetch_data
from utils import aggregate_5min_to_30min_custom

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
api_key = os.getenv('API_KEY')

def convert_from_epoch(epoch: int) -> datetime:
    """
    Convert epoch timestamp to datetime
    Args:
        epoch: Epoch timestamp in seconds
    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(epoch)

def backtest_with_epoch(coin: str, duration_hours: int, epoch_time: int, version: str):
    """Perform backtesting of the LSTM model using epoch timestamp directly"""
    
    try:
        # Step 1: Train the model
        print(f"Training model for {coin} at {convert_from_epoch(epoch_time)}...")
        train_event = {
            'time': epoch_time, 
            'duration': duration_hours,
            'currency': f"{coin}-USD",
            'version': version
        }
        handler(train_event, None)

        # Step 2: Test the model
        print(f"Testing model for {coin}...")
        test_event = json.dumps({
            'requestid': '726d076c-cdb5-43c8-a83d-b706c7bd71e5',
            'command': 'RESULT',
            'actual_payload': f'{coin.lower()}-usd,{duration_hours}',
            'payload': {
                'time': epoch_time,
                'duration': duration_hours,
                'currency': f"{coin}-USD",
                'version': version
            }
        })

        parsed_event = json.loads(test_event)
        predicted_data, input_data = handler_test(parsed_event, None)

        # Step 3: Get Actual Data and show comparison
        print(f"Fetching actual data for {coin}...")

        df_5min = pd.read_excel(os.path.join(os.path.dirname(__file__), "btc_5min_data.xlsx"))
        df_5min['TIMESTAMP'] = df_5min['TIMESTAMP'].astype(int)

        if version in ["v1.1", "v2.1", "v2.2"]:
            epoch_time = (epoch_time // 300) * 300  # Round down to the nearest 5-minute interval
        else:
            epoch_time = (epoch_time // 1800) * 1800

        start_dt_in_epoch = epoch_time
        end_dt_in_epoch = start_dt_in_epoch + (duration_hours * 3600)

        df_5min = df_5min[
            (df_5min['TIMESTAMP'] >= start_dt_in_epoch) &
            (df_5min['TIMESTAMP'] <= end_dt_in_epoch)
        ]

        df_30min = aggregate_5min_to_30min_custom(df_5min)

        if not isinstance(df_30min, pd.DataFrame):
            df_30min = pd.DataFrame(df_30min)

        actual_data = df_30min[['TIMESTAMP', 'CLOSE']].to_dict(orient='records')
        actual_window = df_30min[['TIMESTAMP', 'CLOSE']].copy()

        actual_prices = np.array([entry['CLOSE'] for entry in actual_data])
        predicted_prices = np.atleast_1d(predicted_data)

        min_len = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:min_len]
        predicted_prices = predicted_prices[:min_len]

        if len(actual_prices) == 0 or len(predicted_prices) == 0:
            print(f"No data available for calculation.")
            mape_score = float('nan')
        else:
            mape_score = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        actual_prediction_timestamps = actual_window['TIMESTAMP'].values[:min_len]

        return {
            'timestamp': epoch_time,
            'mape_score': mape_score,
            'actual_prediction_timestamps': actual_prediction_timestamps,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices,
            'status': 'success'
        }

    except Exception as e:
        print(f"Error processing {epoch_time}: {str(e)}")
        return {
            'timestamp': epoch_time,
            'mape_score': float('nan'),
            'actual_prediction_timestamps': [],
            'actual_prices': [],
            'predicted_prices': [],
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main function to process January data only"""
    
    # Configuration
    start_epoch = 1742090391 
    end_epoch = 1742919282
    version = "v2.1"
    duration_hours = 2 

    print("="*80)
    print("JANUARY BACKTEST EXECUTION STARTED - SINGLE PROCESS")
    print("="*80)
    print(f"Version: {version}")
    print(f"Duration: {duration_hours} hours")
    print(f"Interval: 5 minutes")
    print(f"Month: January 2025")
    print(f"Start time: {convert_from_epoch(start_epoch)}")
    print(f"End time: {convert_from_epoch(end_epoch)}")
    print(f"Save frequency: Every iteration (real-time)")
    print("="*80)
    
    results = []
    current_time = start_epoch
    iteration_count = 0
    success_count = 0
    error_count = 0
    
    # Calculate total iterations for progress tracking
    total_iterations = (end_epoch - start_epoch) // 300  # 5-minute intervals
    print(f"Expected total iterations: {total_iterations}")
    print(f"Expected duration: ~{total_iterations * 10 / 3600:.1f} hours (assuming 10 seconds per prediction)")
    print("="*80)
    
    while current_time <= end_epoch:
        iteration_count += 1
        
        print(f"\n🔄 Iteration {iteration_count}/{total_iterations} - Timestamp: {current_time}")
        
        result = backtest_with_epoch(
            coin="BTC",
            duration_hours=duration_hours,
            epoch_time=current_time,
            version=version
        )
        
        # Prepare row for Excel - simplified format (ONLY timestamp and values)
        row = {
            'timestamp': current_time,
            'datetime': convert_from_epoch(current_time).strftime('%Y-%m-%d %H:%M:%S'),
            'status': result.get('status', 'unknown'),
            'error': result.get('error', None)
        }
        
        # Add predicted values
        predicted_prices = result.get('predicted_prices', [])
        for i in range(4):
            if isinstance(predicted_prices, np.ndarray) and i < len(predicted_prices):
                row[f'prediction_value{i+1}'] = float(predicted_prices[i])
            else:
                row[f'prediction_value{i+1}'] = None
        
        # Add actual values
        actual_prices = result.get('actual_prices', [])
        for i in range(4):
            if isinstance(actual_prices, np.ndarray) and i < len(actual_prices):
                row[f'actual_value{i+1}'] = float(actual_prices[i])
            else:
                row[f'actual_value{i+1}'] = None
        
        results.append(row)
        
        # Track success/error counts
        if result.get('status') == 'success':
            success_count += 1
            print(f"✅ Success - MAPE: {result.get('mape_score', 'N/A'):.2f}%")
        else:
            error_count += 1
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        current_time += 300  # Increment by 5 minutes (300 seconds)
        
        # Save at each iteration to prevent data loss
        df_results = pd.DataFrame(results)
        Month="jan"
        excel_filename = f'backtest_results_2_{version}_{duration_hours}h.xlsx'
        from filelock import FileLock

        with FileLock(excel_filename + ".lock"):
            df_results.to_csv(excel_filename, index=False)        
        
        time.sleep(1)
        
        # Print progress every 50 iterations (more frequent since single process)
        if iteration_count % 5 == 0:
            # excel_filename = f'backtest_results_2_{version}_{duration_hours}h.xlsx'
            # df_results.to_excel(excel_filename, index=False)
            # time.sleep(1)
            progress = (iteration_count / total_iterations) * 100
            print(f"\n📊 Progress: {progress:.1f}% ({iteration_count}/{total_iterations})")
            print(f"📈 Success: {success_count}, Errors: {error_count}")
            print(f"💾 Saved {len(results)} rows to {excel_filename}")
            print("-" * 50)

    # excel_filename = f'backtest_results_2_{version}_{duration_hours}h.xlsx'
    # df_results.to_excel(excel_filename, index=False)
    
    # Final summary
    print("\n" + "="*80)
    print("JANUARY EXECUTION COMPLETED")
    print("="*80)
    print(f"Total iterations processed: {iteration_count}")
    print(f"Successful predictions: {success_count}")
    print(f"Failed predictions: {error_count}")
    print(f"Success rate: {(success_count/iteration_count)*100:.1f}%")
    print(f"Final file: {excel_filename}")
    print(f"Total rows saved: {len(results)}")
    
    if success_count > 0:
        df_final = pd.DataFrame(results)
        successful_rows = df_final[df_final['status'] == 'success']
        if len(successful_rows) > 0 and 'prediction_value1' in successful_rows.columns:
            avg_pred = successful_rows['prediction_value1'].mean()
            avg_actual = successful_rows['actual_value1'].mean()
            print(f"Average prediction value 1: ${avg_pred:.2f}")
            print(f"Average actual value 1: ${avg_actual:.2f}")
    
    print("="*80)

if __name__ == "__main__":
    print("Starting January-only backtest execution...")
    print(f"Start epoch: 1735689600 ({convert_from_epoch(1735689600)})")
    print("IMPORTANT: Data will be saved at each iteration to prevent loss!")
    print("No multiprocessing - single sequential execution")
    
    main()