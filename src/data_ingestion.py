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


# def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path : str) -> None:
#   """saving train and test data"""
#   try:
#     raw_data_path = os.path.join(data_path,'raw')
#     os.makedirs(raw_data_path,exist_ok=True)
#     train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
#     test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
#     logger.debug('data is successfully saved to %s',raw_data_path)
#     print("✅ Data ingestion pipeline completed successfully.")
#   except Exception as e:
#     logger.debug('unexpected error occured during saving data: %s', e)
#     raise


def main(raw_data_path='./data'):
  try:
    test_size = 0.2
    data_path = 'https://raw.githubusercontent.com/KHAJA-MUBASHIR-ARSALAN/end-to-end-mlops/main/experiments/Dataset.csv'
    df = load_data(data_url=data_path)
    final_df = preprocessing(df)
    x = final_df.drop("Aggregate rating",axis=1)
    y = final_df['Aggregate rating']
    train_x, test_x, train_y, test_y = tts(x, y, test_size=test_size, random_state=42)

    print("reset index for row alignement after refresh")
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    print("saving tts files into data folder")

    train_x.to_csv(os.path.join(raw_data_path, "train_X.csv"), index=False)
    train_y.to_csv(os.path.join(raw_data_path, "train_y.csv"), index=False)
    test_x.to_csv(os.path.join(raw_data_path, "test_X.csv"), index=False)
    test_y.to_csv(os.path.join(raw_data_path, "test_y.csv"), index=False)


    print(f"Saved files to {os.path.abspath(raw_data_path)}")
    return {
        "train_X": os.path.join(raw_data_path, "train_X.csv"),
        "train_y": os.path.join(raw_data_path, "train_y.csv"),
        "test_X": os.path.join(raw_data_path, "test_X.csv"),
        "test_y": os.path.join(raw_data_path, "test_y.csv"),
    }
  except Exception as e:
    logger.debug('failed to complete the data ingestion process: %s',e)
    print(f'error{e}')


if __name__ == '__main__':
  main()