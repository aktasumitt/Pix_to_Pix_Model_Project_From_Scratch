from torchvision.transforms import transforms
from torch.utils.data import random_split
from src.components.data_transformation.dataset_module import DatasetModule
from src.exception.exception import ExceptionNetwork,sys
from src.entity.config_entity import DataTransformationConfig
from src.utils import load_json,save_obj
from src.logger import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def transformer(self,resize_size:int=286,crop_size:int=256,edge:bool=False):
        try:
            transformer=transforms.Compose([transforms.Resize((resize_size,resize_size)),
                                            transforms.RandomCrop((crop_size,crop_size))
                                            ])
            
            if edge==True:
                transformer=transforms.Compose([transformer,
                                                 transforms.Grayscale(1)])
            
            tensor_transformer=transforms.Compose([transformer,
                                                  transforms.ToTensor()])
            return tensor_transformer
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)
    
    def random_split_data(self,dataset,test_rate,valid_rate):
        try:
            test_size=int(len(dataset)*test_rate)
            valid_size=int(len(dataset)*valid_rate)
            train_Size=len(dataset)-test_size-valid_size
            
            train,valid,test=random_split(dataset,[train_Size,valid_size,test_size])
            
            return train,valid,test
        
        except Exception as e:
            raise ExceptionNetwork(e, sys) 
    
    
    def create_and_save_dataset_transformed(self):
        try:
            
            real_img_list=load_json(path=self.config.real_img_path_list)
            edge_img_list=load_json(path=self.config.edge_img_path_list)
            
            transformer_real = self.transformer(self.config.resize_size,self.config.crop_size)
            transformer_edge = self.transformer(self.config.resize_size,self.config.crop_size,edge=True)
            
            full_dataset = DatasetModule(real_img_list,edge_img_list,transformer_real,transformer_edge)  
            
            train_dataset,valid_dataset,test_dataset=self.random_split_data(full_dataset,
                                                                            test_rate=self.config.test_rate,
                                                                            valid_rate=self.config.valid_rate)
            
            save_obj(train_dataset, save_path=self.config.train_dataset_save_path)
            logger.info(f"train dataset was saved on {self.config.train_dataset_save_path}")
            
            save_obj(valid_dataset, save_path=self.config.valid_dataset_save_path)
            logger.info(f"validation dataset was saved on {self.config.valid_dataset_save_path}")
            
            save_obj(test_dataset, save_path=self.config.test_dataset_save_path)
            logger.info(f"test dataset was saved on {self.config.test_dataset_save_path}")
            
        except Exception as e:
            raise ExceptionNetwork(e, sys)


if __name__ == "__main__":
    config = DataTransformationConfig()
    data_transformation = DataTransformation(config)
    data_transformation.create_and_save_dataset_transformed()
