#%% Imports

import os
import json

import rasterio as rio
from rasterio.enums import Resampling
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip

#%% The BigEarthNet dataset class

class BigEarthNet(Dataset):
    """
    To use this class, you must first download the BigEarthNet dataset by navigating to the "data"-directory and running "bash download_bigearthnet.sh"
    """

    class_label_map = {
        "Continuous urban fabric": 0, # 0: Urban Fabric
        "Discontinuous urban fabric": 0,
        "Industrial or commercial units": 1, # 1: Industrial and Commercial Units
        "Non-irrigated arable land": 2, # 2: Arable Land
        "Permanently irrigated land": 2,
        "Rice fields": 2,
        "Vineyards": 3, # 3: Permanent crops
        "Fruit trees and berry plantations": 3,
        "Olive groves": 3,
        "Pastures": 4, # 4: Pastures
        "Annual crops associated with permanent crops": 3,
        "Complex cultivation patterns": 5, # 5: Complex Cultivation Patterns
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 6, # 6: Land principally occupied by agriculture, with significant areas of natural vegetation
        "Agro-forestry areas": 7, # 7: Agro-forestry areas
        "Broad-leaved forest": 8, # 8: Broad-leaved forest
        "Coniferous forest": 9, # 9: Coniferous forest
        "Mixed forest": 10, # 10: Mixed forest
        "Natural grassland": 11, # 11: Natural grassland and sparsely vegetated areas
        "Moors and heathland": 12, # 12: Moors, heathland and sclerophyllous vegetation
        "Sclerophyllous vegetation": 12,
        "Transitional woodland/shrub": 13, # 13: Transitional woodland, shrub
        "Beaches, dunes, sands": 14, # 14: Beaches, dunes, sands
        "Sparsely vegetated areas": 11,
        "Inland marshes": 15, # 15: Inland wetlands
        "Peatbogs": 15,
        "Salt marshes": 16, # 16: Coastal wetlands
        "Salines": 16,
        "Water courses": 17, # 17: Inland waters
        "Water bodies": 17,
        "Coastal lagoons": 18, # 18: Marine waters
        "Estuaries": 18,
        "Sea and ocean": 18
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
            self.data = [self.__getitem__(idx) for idx in tqdm(range(self.__len__()))]
            self.in_memory = True

    def __len__(self):

        return len(self.df.index)

    def __getitem__(self, idx):

        if self.in_memory:

            x, y, aux = self.data[idx]

        else:

            elem = self.df.loc[idx]

            band_paths = sorted([os.path.join(elem.at['path'], f) for f in os.listdir(elem.at['path']) if f.endswith('.tif')])
            x = [torch.tensor(rio.open(band_path).read(out_dtype="float32", out_shape=(1,120,120), resampling=Resampling.bilinear)) for band_path in band_paths]
            x = torch.concatenate(x, dim=0).float()
            x = torch.clip(x / 10000, 0, 1)
            
            y = torch.zeros((19,))
            metadata_path = [os.path.join(elem.at['path'], f) for f in os.listdir(elem.at['path']) if f.endswith('.json')][0]
            with open(metadata_path) as metadata:
                for label_str in json.load(metadata)["labels"]:
                    try:
                        y[self.class_label_map[label_str]] = 1
                    except KeyError:
                        continue
        
            aux = torch.tensor(elem.filter(regex="worldclim*")).float()

        x = self.transforms(x)

        return x, y, aux

class BigEarthNetLoader(DataLoader):

    def __init__(self, ds, batch_size, shuffle=True):

            super().__init__(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)