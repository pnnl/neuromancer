import os


def set_environment_variables():
    raise AssertionError('Please update MLFLOW secrets in neuromancer/pnl_secrets_mlflow.py')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://dadaist-s3.mlflow.pnl.gov'
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dadaist-server.mlflow.pnl.gov'
    os.environ['AWS_ACCESS_KEY_ID'] = ''
    os.environ['AWS_SECRET_ACCESS_KEY'] = ''
    os.environ['MLFLOW_TRACKING_USERNAME'] = ''
    os.environ['MLFLOW_TRACKING_PASSWORD'] = ''

