from src.components.model.generator import Generator
from src.components.model.discriminator import Discriminator
from src.exception.exception import ExceptionNetwork,sys
from src.utils import save_obj
from src.logger import logger
from src.entity.config_entity import ModelIngestionConfig


class ModelIngestion():
    
    def __init__(self, config: ModelIngestionConfig):
        try:
            self.config = config
            
            self.generator=Generator(config.channel_size_in,config.channel_size_out) # Encoder Model    
            self.discriminator=Discriminator(config.channel_size_out,config.channel_size_in) # Decoder Model
            
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
    def initiate_and_save_model(self):
        try:
            save_obj(self.generator, self.config.generator_save_path)
            save_obj(self.discriminator, self.config.discriminator_save_path)
            
            logger.info("generator ve discriminator modelleri artifacts içerisine kaydedildi")
            
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        
if __name__ == "__main__":
    config = ModelIngestionConfig()
    model_ingestion = ModelIngestion(config)
    model_ingestion.initiate_and_save_model()

