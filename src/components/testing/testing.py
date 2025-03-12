import PIL.Image
from src.entity.config_entity import TestConfig
from src.components.testing.test_module import model_test
from src.utils import save_as_json,load_obj,load_checkpoints,save_obj

from torch.utils.data import DataLoader
import torch
from src.logger import logger
from src.exception.exception import ExceptionNetwork,sys

class Testing():
    def __init__(self,config:TestConfig):
        self.config=config
        
        self.generator=load_obj(self.config.final_model_generator_path)
        self.discriminator= load_obj(self.config.final_model_discriminator_path)
        
        self.loss_bce=torch.nn.BCEWithLogitsLoss()
        self.loss_l1=torch.nn.L1Loss()

    def load_object(self):
        try:
            test_dataset=load_obj(self.config.test_dataset_path)
            test_dataloader=DataLoader(test_dataset,batch_size=self.config.batch_size,shuffle=True)   
            
            return test_dataloader
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
            
    def initiate_testing(self):
        try:
            
            test_dataloader = self.load_object()
            
            # Load Checkpoints if u want
            if self.config.load_checkpoints_for_test==True:
                load_checkpoints(path=self.config.checkpoint_path,
                                model_gen=self.generator,
                                model_disc=self.discriminator)    
            
            fake_grid,test_loss = model_test(test_dataloader=test_dataloader,
                                             model_discriminator=self.discriminator,
                                             model_generator=self.generator,
                                             loss_bce=self.loss_bce,
                                             loss_l1=self.loss_l1)
                            
            save_as_json(data={"Test_loss":test_loss},save_path=self.config.test_result_save_path)
            fake_grid.save(self.config.test_generated_image_path)
            
            logger.info(f"Testing model is completed. Test results was saved")
            
            
            
            # Save tested model if you want (you can use this after load spesific checkpoints)
            if self.config.save_tested_model==True:
                # save final model
                save_obj(self.generator,save_path=self.config.test_model_generator_save_path)
                logger.info(f"Final model is saved on [{self.config.test_model_generator_save_path}]")
                
                # save final model
                save_obj(self.discriminator,save_path=self.config.test_model_discriminator_save_path)
                logger.info(f"Final model is saved on [{self.config.test_model_discriminator_save_path}]")        
    
                
            
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)