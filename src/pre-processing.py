import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ensure the log dir

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


# Logging configuration

logger = logging.getLogger('preprocessing')
logger.setLevel('DEBUG')

# console defining 

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler

log_file_path = os.path.join(log_dir,'preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting up formatter 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



num_cols = ['Average Cost for two','Price range','Aggregate rating']



def preprocessing(df : pd.DataFrame, cols = num_cols) -> pd.DataFrame:
  """preprocessing the data and the columns which are irrelevent deleting it"""
  try:
    missing = [col for col in num_cols if col not in df.columns]
    if missing:
      raise KeyError(f"Missing required numeric columns: {missing}")
    df[num_cols].boxplot(figsize=(10,6))
    plt.title('numeric features before transformation')
    plt.xticks(rotation=45)
    plt.show()


    for col in num_cols:
      lb = df[col].quantile(0.25)
      ub = df[col].quantile(0.75)
      df[col] = np.where(
        df[col]>ub,
        ub,
        np.where(
            df[col] < lb,
            lb,
            df[col]
        )
    )



    df[num_cols].boxplot(figsize=(10,6))
    plt.title('numeric features After transformation')
    plt.xticks(rotation=45)
    plt.show()


    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    df_scaled[num_cols].boxplot(figsize=(10,6))
    plt.title('numeric features After transformation')
    plt.xticks(rotation=45)
    plt.show()


    return df_scaled
  except KeyError as e:
    logger.debug('Missing column in the dataframe: %s',e)
    raise
  except Exception as e:
    logger.debug('unexpected error is occured')
    raise

def main(cols='num_cols'):
  try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.debug('data loaded successfully')

    # process

    train_processed_data = preprocessing(train_data,cols)
    test_processed_data = preprocessing(test_data,cols)

    data_path = os.path.join("./data","interm")
    os.makedirs(data_path,exist_ok=True)

    train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index = False)
    test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index = False)

    logger.debug('data saved successfully: %s',data_path)
  except FileNotFoundError as e:
    logger.error('file not found %s',e)
  except pd.errors.EmptyDataError as e:
    logger.error('no data: %s',e)
  except Exception as e:
    logger.error('failed to complete the preprocessing: %s',e)
    print(f'error{e}')

if __name__ == '__main__':
  main()