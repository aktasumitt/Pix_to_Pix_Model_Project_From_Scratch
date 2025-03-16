from torch.utils.data import Dataset
from PIL import Image


class DatasetModule(Dataset):
    def __init__(self,real_img_list,edge_img_list,transformer_real,transformer_edge):
        super(DatasetModule,self).__init__()
        self.transformer_edge=transformer_edge
        self.transformer_real=transformer_real
        self.real_img_list=real_img_list
        self.edge_img_list=edge_img_list        
    
    def __len__(self):
        return len(self.real_img_list)

    def __getitem__(self, index):

        real_img_path=self.real_img_list[index]
        real_image=Image.open(real_img_path)
        real_image=self.transformer_real(real_image)
        
        edge_img_path=self.edge_img_list[index]
        edge_image=Image.open(edge_img_path)
        edge_image=self.transformer_edge(edge_image)
                
        return (edge_image,real_image)