import pandas as pd
import logging
import os 
from sklearn.model_selection import train_test_split as tts 


# ensure the log dir

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


# Logging configuration

logger = logging.getLogger('data ingestion')
logger.setLevel('DEBUG')

# console defining 

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler

log_file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting up formatter 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



# loading data 


def load_data(data_url : str) -> pd.DataFrame:
  """Loading data"""
  try:
    df = pd.read_csv(data_url)
    return df
  except pd.errors.ParserError as e:
    logger.error('failed to parse the csv file %s',e)
    raise
  except Exception as e:
    logger.error('unexpected error occure while loading the data %s',e)
    raise


# df.drop(["Restaurant ID","Votes", "Country Code","Latitude","Longitude","Restaurant Name","Address","Locality","Locality Verbose"],axis = 1, inplace = True)


def preprocessing(df : pd.DataFrame) -> pd.DataFrame:
  """preprocessing the data and the columns which are irrelevent deleting it"""
  try:
    df.drop(columns=["Restaurant ID","Votes", "Country Code","Latitude","Longitude","Restaurant Name","Address","Locality","Locality Verbose"],axis = 1, inplace = True)
    logger.debug('Data processing completed')
    print('preprocessing done')
    return df
  except KeyError as e:
    logger.debug('Missing column in the dataframe: %s',e)
    raise
  except Exception as e:
    logger.debug('unexpected error is occured %s',e)
    raise


def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path : str) -> None:
  """saving train and test data"""
  try:
    raw_data_path = os.path.join(data_path,'raw')
    os.makedirs(raw_data_path,exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
    test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
    logger.debug('data is successfully saved to %s',raw_data_path)
    print("✅ Data ingestion pipeline completed successfully.")
  except Exception as e:
    logger.debug('unexpected error occured during saving data: %s', e)
    raise


def main():
  try:
    test_size = 0.2
    data_path = 'https://raw.githubusercontent.com/KHAJA-MUBASHIR-ARSALAN/end-to-end-mlops/main/experiments/Dataset.csv'
    df = load_data(data_url=data_path)
    final_df = preprocessing(df)
    train_data, test_data = tts(final_df,test_size=test_size,random_state=42)
    save_data(train_data,test_data, data_path='./data')
    logger.debug('data fetch succesfully %s',data_path)
  except Exception as e:
    logger.debug('failed to complete the data ingestion process: %s',e)
    print(f'error{e}')


if __name__ == '__main__':
  main()