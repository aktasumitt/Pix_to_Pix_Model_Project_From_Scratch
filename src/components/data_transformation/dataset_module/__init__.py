from torch.utils.data import Dataset
from PIL import Image


class DatasetModule(Dataset):
    def __init__(self,real_img_list,edge_img_list,transformer_real,transformer_edge):
        super(DatasetModule,self).__init__()
        self.transformer_edge=transformer_edge
        self.transformer_real=transformer_real
        self.real_img_list=real_img_list
        self.edge_img_list=edge_img_list
        
        
    def get_img(self,img_list,index,transformer):
        img_path=img_list[index]
        image=Image.open(img_path).convert("RGB")
        image=transformer(image)
        return image
    
    def __len__(self):
        return len(self.real_img_list)

    def __getitem__(self, index):
        
        real_img=self.get_img(self.real_img_list,index,self.transformer_real)
        edge_img=self.get_img(self.edge_img_list,index,self.transformer_edge)
                
        return (edge_img,real_img)