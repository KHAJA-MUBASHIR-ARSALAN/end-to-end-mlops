import os
import pandas as pd
import logging
import numpy as np
from data_ingestion import main as ingest_main



# ensure the log dir

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


# Logging configuration

logger = logging.getLogger('featur-engineering')
logger.setLevel('DEBUG')

# console defining 

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler

log_file_path = os.path.join(log_dir,'feature-engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting up formatter 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def feature_engineering(df : pd.DataFrame, col : str = 'City') -> pd.DataFrame:
  """feature engineering the data and the columns which are irrelevent deleting it"""
  try:
    df[col] = df[col].astype(str).apply(lambda x: x.split(',')[0].strip())
    print('feature engineering done')
    logger.debug('feature engineering completed')
    return df
  except KeyError as e:
    logger.error('failed to apply the changes in column %s',e)
    raise
  except Exception as e:
    logger.error('unexpected error occure while feature engineering the data %s',e)
    raise



def run():
    data = ingest_main()   # returns dict with paths or dfs
    print("Keys returned by ingest_main():", data.keys())
    # Example: {'train_X': 'data/raw/train_X.csv', ...}

    # If values are file paths (strings):
    train_x_path = data['train_X']
    print("train_X path:", train_x_path)

    train_x_df = pd.read_csv(train_x_path)
    print("Columns in train_x_df:", train_x_df.columns)

    # Now pass the REAL DataFrame to feature_engineering
    df_fe = feature_engineering(train_x_df, 'City')
    print(df_fe.head())


if __name__ == '__main__':
  run()