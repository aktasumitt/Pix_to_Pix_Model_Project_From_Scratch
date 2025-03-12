from src.logger import logger
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path
from src.utils import save_as_json


class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # Loading Dataset
    def loading_Dataset(self,dataset_dir):
        try:  
            img_path_list=[]
            
            for i,folder in enumerate(Path(dataset_dir).glob("*")):
                for _,img in enumerate(Path(folder).glob("*")):
                    img_path_list.append(str(img))
                
                if i==3: break
                    
            logger.info(f"...{len(img_path_list)} data were loaded")    
            return img_path_list
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            real_image_path_list=self.loading_Dataset(dataset_dir=self.config.local_real_data_path)
            save_as_json(real_image_path_list,save_path=self.config.save_real_data_path)
            
            edge_image_path_list=self.loading_Dataset(dataset_dir=self.config.local_edge_data_path)
            save_as_json(edge_image_path_list,save_path=self.config.save_edge_data_path)
            
            logger.info(f"data path lists were saved in artifacts folder")
            
        except Exception as e:
            raise ExceptionNetwork(e,sys)

if __name__=="__main__":
    
    
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()