import json
import os
import yaml
import pickle
import torch
from box import ConfigBox
from pathlib import Path
from src.exception.exception import ExceptionNetwork,sys

def save_as_json(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path,"w") as f:
            json.dump(data,f)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)

def load_json(path:Path):
    try:
        path=Path(path)
        with open(path, "r") as f:
            loaded_data = json.load(f)
        
        return loaded_data
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_as_yaml(data,save_path:Path):
    try:
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, allow_unicode=True)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def load_yaml(path:Path):  
    try:     
        path=Path(path)
        with open(f"{path}", "r", encoding="utf-8") as file:
            loaded_dict = yaml.safe_load(file) 
        
        return ConfigBox(loaded_dict)
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_obj(data,save_path:Path):
    try:    
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        save_path=Path(save_path)
        
        with open(save_path, "wb") as file:
            torch.save(data, file)
            
    except Exception as e:
        raise ExceptionNetwork(e,sys)
        
        
def load_obj(path:Path):
    try:    
        path=Path(path)
        with open(path, "rb") as file:
            obj=torch.load(file,map_location=torch.device("cpu"),weights_only=False)
            
        return obj
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def save_checkpoints(save_path,epoch,model_gen,model_disc,optimizer_gen,optimizer_disc):
    try:    
        checkpoints_name=Path(f"checkpoint_{epoch}-epoch.pth.tar") # to save every_epoch
        
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        
        
        epoch_path=os.path.join(save_dir,checkpoints_name)
        latest_path=Path(save_path)
        
        checkpoint={"Epoch":epoch,
                    "Optimizer_Disc_State":optimizer_disc.state_dict(),
                    "Optimizer_Gen_State":optimizer_gen.state_dict(),
                    "Model_Gen_State":model_gen.state_dict(),
                    "Model_Disc_State":model_disc.state_dict()}
        
        torch.save(checkpoint,f=latest_path)
        torch.save(checkpoint,f=epoch_path)
        
    except Exception as e:
        raise ExceptionNetwork(e,sys)


def load_checkpoints(model_gen,model_disc,path:Path,optimizer_gen=None,optimizer_disc=None)->int:
    try:    
        path=Path(path)
        
        checkpoint=ConfigBox(torch.load(path))
        model_gen.load_state_dict(checkpoint.Model_Gen_State)
        model_disc.load_state_dict(checkpoint.Model_Disc_State)
        
        if optimizer_gen!=None and optimizer_disc!=None:
            start_epoch=checkpoint.Epoch
            optimizer_gen.load_state_dict(checkpoint.Optimizer_Gen_State)
            optimizer_disc.load_state_dict(checkpoint.Optimizer_Disc_State)
        
            return start_epoch+1
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)
    

    