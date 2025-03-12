import torch
from torchvision.utils import make_grid
from src.exception.exception import ExceptionNetwork,sys
import numpy as np
import PIL.Image
import tqdm



def validation_model(valid_dataloader,model_generator,model_discriminator,loss_bce,loss_l1,device,LAMDA=10):  # Generator
        
        try:    
                loss_total=0
                pb=tqdm.tqdm(range(len(valid_dataloader)),"Validation Progress")
                
                # We will see generated fake images for each labels 5 times
                for batch,(edge_img,real_img) in enumerate(valid_dataloader):
                
                        model_generator.eval()
                        model_discriminator.eval()
                        
                        edge_img=edge_img.to(device)
                        real_img=real_img.to(device)
                                        
                        with torch.no_grad():
                                fake_img=model_generator(edge_img)
                                fake_disc_gen=model_discriminator(edge_img,fake_img)
                
                                gen_loss_bce=loss_bce(fake_disc_gen,torch.ones_like(fake_disc_gen))
                                l1_loss=loss_l1(fake_img,real_img)
                                
                                loss_gen=gen_loss_bce+(LAMDA*l1_loss)

                        loss_total+=loss_gen.item()
                        pb.update(1)
                
                pb.set_postfix({"loss_valid":loss_total/(batch+1)})
                # Tensorları grid haline getir (nrow belirterek sütun sayısını ayarlıyoruz)
                grid_tensor = make_grid(fake_img, nrow=10, normalize=True, scale_each=True)

                # Grid görüntüsünü NumPy formatına çevir
                grid_np = grid_tensor.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                grid_np = (grid_np * 255).astype(np.uint8)  # Normalizasyonu kaldır, uint8 formatına çevir

                # NumPy array’i PIL Image'e çevir
                grid_pil = PIL.Image.fromarray(grid_np)

        
                return grid_pil,(loss_total/(batch+1))
                
        except Exception as e:
            raise ExceptionNetwork(e,sys)