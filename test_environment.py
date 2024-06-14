from src.utils import get_data
from src.components.data_ingestion import inisiate_data_ingestion
from src.components.data_transformation import inisiate_data_transformation


if __name__=="__main__":
    # Inisiate data ingestion
    train_data_path,test_data_path=inisiate_data_ingestion()
    
    # Inisiaiate data_transformation
    train_array,test_array,_=inisiate_data_transformation(train_path=train_data_path,test_path=test_data_path)