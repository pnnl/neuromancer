import os


def set_environment_variables():
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://dadaist-s3.mlflow.pnl.gov'
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dadaist-server.mlflow.pnl.gov'
    os.environ['AWS_ACCESS_KEY_ID'] = 'GYOlJV8ZON'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'HvxsMuQtnHQoI9nsBkyJ3Yfm3TEyyZ5M8vq8UJGY'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'harall-microll'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'z47A1OsL2wFN hxVSa6tXqRo4 WNXc2W9YuZvq'

