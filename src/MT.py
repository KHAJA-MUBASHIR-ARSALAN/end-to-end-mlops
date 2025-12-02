import os
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler,LabelEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib


# ensure the log dir

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


# Logging configuration

logger = logging.getLogger('model-training')
logger.setLevel('DEBUG')

# console defining 

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler

log_file_path = os.path.join(log_dir,'model-training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# setting up formatter 

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
print("logging configured for model training")


def build_preprocessor():
  try:
    logger.debug('building preprocessor started')
    ordinal_pipe = Pipeline([
    ('ordinal_',OrdinalEncoder())
    ])

    onehot_pipe = Pipeline([
    ('onehot_',OneHotEncoder(sparse_output = False,drop = 'first',handle_unknown = 'ignore'))
    ])

    num_pipe = Pipeline([
    ('minmax_',MinMaxScaler())
    ])

    preprocessing = ColumnTransformer([
    ('ordinal',ordinal_pipe,['Price range', 'Rating text', 'Rating color','Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']),
    ('onehot',onehot_pipe,['Currency', 'City']),
    ('num',num_pipe, ['Average Cost for two'])
    ],remainder = 'passthrough')
    logger.debug('building preprocessor completed')
    return preprocessing
    

  except Exception as e:
    logger.error('unexpected error occure while building preprocessor %s',e)
    raise




# buildng pipeline


def build_model_pipeline(model):
  try:
    logger.debug('building model pipeline started')
    preprocessor = build_preprocessor()

    pipe = Pipeline([
    ('preprocessor',preprocessor),
    ('model',model)
    ])
    logger.debug('building model pipeline completed')
    return pipe
  except Exception as e:
    logger.error('unexpected error occure while building model pipeline %s',e)
    raise


def build_model(model_class, **prams):
  try:
    logger.debug('building model started')
    model = model_class(**prams)
    logger.debug('building model completed')
    return model
  except Exception as e:
    logger.error('unexpected error occure while building model %s',e)
    raise



def train():
  logger.debug('model training started')
  train_x = pd.read_csv('./data/interm/train_x_processed_data.csv')
  train_y = pd.read_csv('./data/interm/train_y_processed_data.csv')

  model = build_model(
        RandomForestRegressor,
        n_estimators=200,
        random_state=42
  )
  build_pipeline = build_model_pipeline(model)

  build_pipeline.fit(train_x,train_y)

  os.makedirs('models',exist_ok=True)
  joblib.dump(build_pipeline,'models/model_pipeline.pkl')
  logger.debug('model training completed and model saved to models/model_pipeline.pkl')



def main():
    try:
        logger.debug("model training main() started")
        train()
        logger.debug("model training main() completed successfully")
    except Exception as e:
        logger.error("model training failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()