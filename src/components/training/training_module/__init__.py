import torch 
import tqdm
from src.exception.exception import ExceptionNetwork,sys


class TrainingModule():
    def __init__(self, train_dataloader, model_generator, model_disc, optimizer_gen, optimizer_disc,
                 loss_bce,loss_l1,gradient_scaler_disc,gradient_scaler_gen, batch_size, device,LAMDA=10):

        self.train_dataloader = train_dataloader
        self.model_generator = model_generator
        self.model_discriminator = model_disc
        self.gradient_scaler_disc=gradient_scaler_disc
        self.gradient_scaler_gen=gradient_scaler_gen
        self.loss_bce=loss_bce
        self.loss_l1=loss_l1
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.batch_size = batch_size
        self.device = device
        self.LAMDA = LAMDA
        
        
        self.model_generator.train()
        self.model_discriminator.train()

    def train_discriminator(self,edge_img,real_img):
        try:

            fake_img=self.model_generator(edge_img)   # Generate Fake img
                
            # Discriminator
            with torch.amp.autocast("cuda"):
                self.model_discriminator.zero_grad()
                fake_disc=self.model_discriminator(edge_img,real_img) 
                real_disc=self.model_discriminator(edge_img,fake_img.detach())


                loss_real=self.loss_bce(real_disc,torch.ones_like(real_disc))
                loss_fake=self.loss_bce(fake_disc,torch.zeros_like(fake_disc))
                loss_discriminator=(loss_real+loss_fake)
                
                self.gradient_scaler_disc.scale(loss_discriminator).backward()
                self.gradient_scaler_disc.step(self.optimizer_disc)
                self.gradient_scaler_disc.update()

            return loss_discriminator.item(),fake_img
        
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    def train_generator(self,edge_img,fake_img,real_img):  # Generator
        try:  
            with torch.amp.autocast("cuda"):
                self.model_generator.zero_grad()
                fake_disc_gen=self.model_discriminator(edge_img,fake_img)
                
                gen_loss_bce=self.loss_bce(fake_disc_gen,torch.ones_like(fake_disc_gen))
                l1_loss=self.loss_l1(fake_img,real_img)
                
                loss_gen=gen_loss_bce+(self.LAMDA*l1_loss)
                
                self.gradient_scaler_gen.scale(loss_gen).backward()
                self.gradient_scaler_gen.step(self.optimizer_gen)
                self.gradient_scaler_gen.update()
                
            return loss_gen.item()
        
        except Exception as e:
                raise ExceptionNetwork(e,sys)
    
    def batch_training(self):
        try:
            PB = tqdm.tqdm(range(len(self.train_dataloader)), "Training Process")
            GEN_LOSS=0
            DISC_LOSS=0

            for batch, (edge_img, real_img) in enumerate(self.train_dataloader):
                
                real_img = real_img.to(self.device)                
                edge_img = edge_img.to(self.device)

                
                # train generator and discriminator
                loss_disc,fake_img=self.train_discriminator(edge_img,real_img)
                loss_gen=self.train_generator(edge_img,fake_img,real_img)
                
                # adding losses for each batch
                GEN_LOSS+=loss_gen
                DISC_LOSS+=loss_disc
                PB.update(1)
                if batch>0 and batch%100==0:
                    PB.set_postfix({"Gen_loss":GEN_LOSS/(batch+1),
                                    "disc_loss":DISC_LOSS/(batch+1)})
                    
            
            TOTAL_GEN_LOSS=GEN_LOSS/(batch+1)
            TOTAL_DISC_LOSS=DISC_LOSS/(batch+1)
                
            return TOTAL_GEN_LOSS,TOTAL_DISC_LOSS
        except Exception as e:
            raise ExceptionNetwork(e,sys)