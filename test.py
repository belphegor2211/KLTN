import pickle
import glob
from PIL import Image
from data.dataset import TextDataset, TextDatasetval
import torch
from models.model import TRGAN
from models.model import Discriminator
import torch


batch_size = 8

TextDatasetObjval = TextDataset()
datasetval = torch.utils.data.DataLoader(
            TextDatasetObjval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True, drop_last=True,
            collate_fn=TextDatasetObjval.collate_fn)

x = 0
for i,data in enumerate(datasetval): 
    print(data['img'].detach().shape)
    if x==10:
        break
    x+=1