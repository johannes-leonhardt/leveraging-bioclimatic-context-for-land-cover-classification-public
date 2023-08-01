#%% Imports

import os

import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.classification import multiclass_accuracy
import wandb

from models.resnet18_encoders import *
from models.classifier import Classifier
from models.training_modules import SupervisedTrainingModule
from models.utils import EarlyStopper

from data.eurosat import EuroSAT, EuroSATLoader

#%% Settings and Hyperparameters

# General settings
N_CHANNELS = 13
N_CLASSES = 10
N_AUX = 20
MAX_EPOCHS = 100
N_RUNS = 10

# Hyperparameters
hyperparameters = {
    "batch_size": 64,
    "learning_rate": 1e-5
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#%% Datasets and -loaders

df = gpd.read_file(os.path.join("data", "eurosat.gpkg"))

# Training, validation, and testing datasets
ds_train = EuroSAT(df[df.split == 'train'])
ds_val = EuroSAT(df[df.split == 'val'])
ds_test = EuroSAT(df[df.split == 'test'], augmentation=False)


# Individual testing datasets for each biogeo. region
bg_dss_test = [
    EuroSAT(df[(df.split == 'test') & (df.biogeo_label.isin(["alpine"]))], augmentation=False),
    EuroSAT(df[(df.split == 'test') & (df.biogeo_label.isin(["arctic", "boreal"]))], augmentation=False),
    EuroSAT(df[(df.split == 'test') & (df.biogeo_label.isin(["atlantic"]))], augmentation=False),
    EuroSAT(df[(df.split == 'test') & (df.biogeo_label.isin(["blackSea", "macaronesia", "mediterranean", "steppic"]))], augmentation=False),
    EuroSAT(df[(df.split == 'test') & (df.biogeo_label.isin(["continental", "pannonian"]))], augmentation=False),
]

print(f"#Samples: Training: {len(ds_train)}, Validation: {len(ds_val)}, Testing: {len(ds_test)} ({[len(bg_ds_test) for bg_ds_test in bg_dss_test]}).")

# Dataloaders
dl_train = EuroSATLoader(ds_train, batch_size=hyperparameters["batch_size"])
dl_val = EuroSATLoader(ds_val, batch_size=hyperparameters["batch_size"])
dl_test = EuroSATLoader(ds_test, batch_size=hyperparameters["batch_size"])
bg_dls_test = [EuroSATLoader(bg_ds_test, batch_size=hyperparameters["batch_size"]) for bg_ds_test in bg_dss_test]

#%% Model settings

# Encoder
ENCODER = ResNet18EncoderEmb
ENCODER_ARGS = [N_CHANNELS, N_AUX]
NAME = "ResNet18_emb"

# Classifier
CLASSIFIER = Classifier
CLASSIFIER_ARGS = [512, N_CLASSES] # First argument needs to dim of latent space

# Path of initial encoder weights from pre-training. Set to None for random initialization.
ENCODER_PATH = None
MODE = "Supervised" if ENCODER_PATH is None else "Downstream"

#%% Training

for i in range(N_RUNS):

    # Initialize model
    encoder = ENCODER(*ENCODER_ARGS).to(device)
    if MODE == "Downstream":
        checkpoint = torch.load(ENCODER_PATH)
        encoder.load_state_dict(checkpoint["encoder"])
    classifier = CLASSIFIER(*CLASSIFIER_ARGS).to(device)
    model = nn.ModuleList([encoder, classifier])

    # Initialize Weights and Biases
    name = f"ES_Mode_{MODE}_Model_{NAME}_{i}"
    wandb.init(
        project = "BGLC-EuroSAT-PS",
        name = name,
        config = hyperparameters,
    )
    save_path = os.path.join("params", name + ".pt")

    # Initialize loss, optimizer, trainer, and early stopper
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    trainer = SupervisedTrainingModule(encoder, classifier, loss_func, optimizer)
    early_stopper = EarlyStopper(patience=5, maximize=True)

    # Training/validation loop
    for _ in range(MAX_EPOCHS):
        # Training
        trainer.training_epoch(dl_train)
        # Validation
        y_hat, y = trainer.predict(dl_train)
        train_acc = multiclass_accuracy(y_hat, y, num_classes=N_CLASSES)
        y_hat, y = trainer.predict(dl_val)
        val_acc = multiclass_accuracy(y_hat, y, num_classes=N_CLASSES)
        # Log state and stop if converged
        wandb.log({"train_acc":train_acc, "val_acc":val_acc})
        if early_stopper.early_stop(val_acc, checkpoint={"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}, save_path=save_path):
            break

    # Reload best performing model
    checkpoint = torch.load(save_path)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])

    # Compute metrics on test set
    y_hat, y = trainer.predict(dl_test)
    test_acc = multiclass_accuracy(y_hat, y, num_classes=N_CLASSES)
    test_acc_bg = []
    for bg_dl_test in bg_dls_test:
        y_hat, y = trainer.predict(bg_dl_test)
        test_acc_bg.append(multiclass_accuracy(y_hat, y, num_classes=N_CLASSES))
    
    # Log results
    wandb.log({
        "test_acc": test_acc,
        "test_acc_alp": test_acc_bg[0],
        "test_acc_bor": test_acc_bg[1],
        "test_acc_atl": test_acc_bg[2],
        "test_acc_med": test_acc_bg[3],
        "test_acc_con": test_acc_bg[4],
    })
    wandb.finish()