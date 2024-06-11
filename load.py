import pickle
import boto3
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
logger = logging.getLogger(__name__)


# Set up logging configuration
log_dir = Path("log_data")
log_dir.mkdir(exist_ok=True)  # Create log directory if it doesn't exist

logging.basicConfig(level=logging.INFO,
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(log_dir / "load.log")
                    ],
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

def load_model(model_name: str, path: str, artifact_dir: str):
    '''
    Load model from S3
    '''
    logger.info(f"Loading {model_name} model from {path}")
    print("loading model")
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    else:
        return "loaded"
    
    path_local_model = os.path.join(artifact_dir, f"{model_name}.pkl")

    if not os.path.exists(path_local_model):
        logger.info("Downloading the model from s3 and then loading.")
        get_model(path, path_local_model)

    with open(path_local_model, "rb") as model_file:
        model = pickle.load(model_file)
        return model

def get_model(path: str, local_path: str):
    '''
    The helper function for load_model
    Args:
        path: The S3 path.
        local_path: The local save path.
    '''
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("REGION_NAME"),
        )

        bucket_name = os.getenv("S3_BUCKET_NAME", default="iyq5197-cloud")
        file_name = path.replace("s3://", "").split("/", 1)[1]

        s3.download_file(bucket_name, file_name, local_path)
        logger.info(f"File downloaded from s3: {path} to the local:{local_path}")
        print("Model download successful.")
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise

def main():
    # Define your models and their respective S3 paths
    models = {
        "model_lasso": "s3://iyq5197-cloud/model_lasso.pkl",
        "model_logistic": "s3://iyq5197-cloud/model_logistic.pkl"
    }
    artifact_dir = "./artifacts"

    # Load each model
    for model_name, s3_path in models.items():
        model = load_model(model_name, s3_path, artifact_dir)
        if model == "loaded":
            logger.info(f"{model_name} model already loaded.")
        else:
            logger.info(f"{model_name} model loaded successfully.")

if __name__ == "__main__":
    main()
