import pandas as pd
import logging
import os 
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import numpy as np

# ensure the log dir

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


# Logging configuration

logger = logging.getLogger('model-evaluation')
logger.setLevel('DEBUG')

# console defining 

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler

log_file_path = os.path.join(log_dir,'model-evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting up formatter 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



# loading data 


def load_model(file_path : str):
  """Loading data"""
  try:
    logger.debug('loading model from %s',file_path)
    with open(file_path, 'rb') as f:
      model = joblib.load(f)
    logger.debug('model loaded from %s',file_path)
    return model
  except pd.errors.ParserError as e:
    logger.error('failed to parse the pkl file %s',e)
    raise
  except Exception as e:
    logger.error('unexpected error occure while loading the data %s',e)
    raise


def evaluate_model(model, test_x_processed_data: np.ndarray, test_y_processed_data: np.ndarray) -> dict:
  """Evaluate the model and return the accuracy"""
  try:
    logger.debug('model evaluation started')
    y_pred = model.predict(test_x_processed_data)
    mse = mean_squared_error(test_y_processed_data, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y_processed_data, y_pred)
    r2 = r2_score(test_y_processed_data, y_pred)

    metrics_dict = {
      'mse': mse,
      'rmse': rmse,
      'mae': mae,
      'r2': r2
    }
    logger.debug('model evaluation completed')
    return metrics_dict
  except Exception as e:
    logger.error('unexpected error occure while evaluating the model %s',e)
    raise

def save_metrics(metrics: dict, file_path: str) -> None:
  """Save evaluation metrics to a file"""
  try:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path, 'w') as f:
      json.dump(metrics, f, indent=4)
    logger.debug('metrics saved to %s',file_path)
  except Exception as e:
    logger.error('unexpected error occure while saving the metrics %s',e)
    raise


def main():
  try:
    logger.debug("model evaluation main() started")
    model =load_model('models/model_pipeline.pkl')
    test_x = pd.read_csv('./data/interm/test_x_processed_data.csv')
    test_y = pd.read_csv('./data/interm/test_y_processed_data.csv').iloc[:,0]
    metrics = evaluate_model(model,test_x,test_y)
    save_metrics(metrics,'reports/metrics.json')
    logger.debug("model evaluation main() completed successfully")
  except Exception as e: 
    logger.error("model evaluation failed: %s", e, exc_info=True)
    raise

if __name__ == '__main__':
  main()