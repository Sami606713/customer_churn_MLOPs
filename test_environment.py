from src.utils import get_data
from src.components.data_ingestion import inisiate_data_ingestion


if __name__=="__main__":
    # Inisiate data ingestion
    train_data,test_data=inisiate_data_ingestion()