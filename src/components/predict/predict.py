import torch
import matplotlib.pyplot as plt
from src.utils import load_obj
from src.entity.config_entity import PredictionConfig
from src.exception.exception import ExceptionNetwork, sys
from pathlib import Path
import os
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image

class PredictionModule():
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.transformer=transforms.Compose([transforms.Resize((self.config.img_size, self.config.img_size)),
                                             transforms.ToTensor(),
                                            #  transforms.Normalize((0.5,), (0.5,))
                                             ])

    def load_image(self, img_dir: Path):
        img_list=[]
        
        for path in Path(img_dir).glob("*"):
            image=Image.open(path).convert("RGB")
            transformed_img=self.transformer(image)
            img_list.append(transformed_img)
        
        return torch.stack(img_list)

    def predict_model(self, model_encoder, model_decoder, model_latent, image_tensor):
        try:
            os.makedirs(self.config.predicted_img_save_path, exist_ok=True)

            model_encoder.eval()
            model_decoder.eval()
            model_latent.eval()
            with torch.no_grad():
                out_encoder = model_encoder(image_tensor)
                out_latent, _ = model_latent(out_encoder)
                out_decoder = model_decoder(out_latent)
                
            for i,img in enumerate(out_decoder):
                
                img_path = os.path.join(self.config.predicted_img_save_path, f"prediction_{i}.jpg")
                save_image(img,img_path)
                

        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def initiate_predict(self):
        
        dataset=self.load_image(img_dir=self.config.prediction_img_load_path)
        encoder_model = load_obj(path=self.config.encoder_path)
        decoder_model = load_obj(path=self.config.decoder_path)
        latent_model = load_obj(path=self.config.latent_path)

        return self.predict_model(model_encoder=encoder_model, model_decoder=decoder_model,
                                  model_latent=latent_model, image_tensor=dataset)
