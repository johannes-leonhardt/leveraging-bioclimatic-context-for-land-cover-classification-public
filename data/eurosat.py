#%% Imports

import numpy as np
import rasterio as rio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip

#%% The EuroSAT dataset class

class EuroSAT(Dataset):
    """
    To use this class, you must first download the EuroSAT dataset by navigating to the "data"-directory and running "bash download_eurosat.sh"
    """

    class_label_map = {
        "AnnualCrop": 0,
        "Forest": 1,
        "HerbaceousVegetation": 2,
        "Highway": 3,
        "Industrial": 4,
        "Pasture": 5,
        "PermanentCrop": 6,
        "Residential": 7,
        "River": 8,
        "SeaLake": 9
    }

    biogeo_label_map = {
        "alpine": 0,
        "arctic": 1,
        "atlantic": 2,
        "blackSea": 3,
        "boreal": 1,
        "continental": 4,
        "macaronesia": 3,
        "mediterranean": 3,
        "pannonian": 4,
        "steppic": 3,
    }

    def __init__(self, df, augmentation=True, in_memory=True):
        
        super().__init__()

        self.df = df.reset_index(drop=True)

        if augmentation:
            self.transforms = torch.nn.Sequential(
                RandomVerticalFlip(0.5),
                RandomHorizontalFlip(0.5)
            )
        else:
            self.transforms = torch.nn.Identity()

        self.in_memory = False
        if in_memory:
            self.data = [self.__getitem__(idx) for idx in range(self.__len__())]
            self.in_memory = True

    def __len__(self):

        return len(self.df.index)

    def __getitem__(self, idx):

        if self.in_memory:
            x, y, aux = self.data[idx]
        
        else:
            elem = self.df.loc[idx]
            x = torch.tensor(rio.open(elem.at["tif_path"]).read(out_dtype="float64")).float()
            x = torch.clip(x / 10000, 0, 1)
            y = torch.tensor(self.class_label_map[elem.at["class_label"]])
            aux = torch.tensor(elem.filter(regex="worldclim*")).float() # These are already normalized

        x = self.transforms(x)

        return x, y, aux

#%% The EuroSAT data loader

class EuroSATLoader(DataLoader):

    def __init__(self, ds, batch_size, shuffle=True):

            super().__init__(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

#%% Method for normalizing images for visualization purposes

def norm_for_visualization(img):

    img_norm = torch.zeros_like(img)

    for i in range(img.shape[0]):
        channel = img[i,:,:]
        q_01 = torch.quantile(channel, 0.01)
        q_99 = torch.quantile(channel, 0.99)
        C_norm = (channel - q_01) / (q_99 - q_01)
        img_norm[i,:,:] = np.clip(C_norm, 0, 1)

    return img_norm