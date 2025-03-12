from torchvision.utils import make_grid
import torch
import tqdm
from src.exception.exception import ExceptionNetwork, sys
from PIL import Image

# Create Pil Image from tensor batch


def create_pil_image(img):
    try:

        x = make_grid(img, 10, scale_each=True, normalize=True)
        x = (x*255).type(torch.uint8)

        # Grid görüntüsünü NumPy formatına çevir
        grid_np = x.cpu().detach().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

        # NumPy array’i PIL Image'e çevir
        grid_pil = Image.fromarray(grid_np)

        return grid_pil

    except Exception as e:
        raise ExceptionNetwork(e, sys)


# Validation Module
def model_test(test_dataloader, model_generator, model_discriminator, loss_bce, loss_l1, LAMDA=10):
    try:
        model_generator.eval()
        model_discriminator.eval()
        loss_total=0

        pb = tqdm.tqdm(range(len(test_dataloader)), "Test progress")

        # We will see generated fake images for each labels 5 times
        for batch, (edge_img, real_img) in enumerate(test_dataloader):

            model_generator.eval()
            model_discriminator.eval()

            edge_img = edge_img
            real_img = real_img

            with torch.no_grad():
                fake_img = model_generator(edge_img)
                fake_disc_gen = model_discriminator(edge_img, fake_img)

                gen_loss_bce = loss_bce(fake_disc_gen, torch.ones_like(fake_disc_gen))
                l1_loss = loss_l1(fake_img, real_img)

                loss_gen = gen_loss_bce+(LAMDA*l1_loss)

                loss_total += loss_gen.item()

            pb.update(1)
        
        pb.set_postfix({"loss_test":loss_total/(batch+1)})
        
        grid_pil=create_pil_image(fake_img)
        
        return grid_pil,loss_total
    
    except Exception as e:
        raise ExceptionNetwork(e, sys)
