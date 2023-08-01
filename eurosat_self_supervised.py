#%% Imports

import os

import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import structural_similarity_index_measure
import wandb

from models.resnet18_encoders import *
from models.resnet18_decoders import *
from models.training_modules import SelfSupervisedTrainingModule
from models.utils import EarlyStopper, MixedAELoss

from data.eurosat import EuroSAT, EuroSATLoader, norm_for_visualization

#%% Settings and Hyperparameters

# General settings
N_CHANNELS = 13
N_CLASSES = 10
N_AUX = 20
MAX_EPOCHS = 200
N_RUNS = 10

# Hyperparameters
hyperparameters = {
    "batch_size": 64,
    "learning_rate": 1e-5,
    "ssim_weight": 0.1
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#%% Datasets and -loaders

df = gpd.read_file(os.path.join("data", "eurosat.gpkg"))

# Training, validation, and testing datasets
ds_train = EuroSAT(df[df.split == 'train'])
ds_val = EuroSAT(df[df.split == 'val'])
ds_test = EuroSAT(df[df.split == 'test'], augmentation=False)

print(f"#Samples: Training: {len(ds_train)}, Validation: {len(ds_val)}, Testing: {len(ds_test)}.")

# Dataloaders
dl_train = EuroSATLoader(ds_train, batch_size=hyperparameters["batch_size"])
dl_val = EuroSATLoader(ds_val, batch_size=hyperparameters["batch_size"])
dl_test = EuroSATLoader(ds_test, batch_size=hyperparameters["batch_size"])

#%% Model settings

# Encoder
ENCODER = ResNet18EncoderCbn
ENCODER_ARGS = [N_CHANNELS, N_AUX]
ENCODER_NAME = "ResNet18_cbn"

# Decoder
DECODER = ResNet18DecoderCbn
DECODER_ARGS = [N_CHANNELS, N_AUX]
DECODER_NAME = "ResNet18_cbn"

MODE = "Pretext"

#%% Training

for i in range(N_RUNS):

    # Initialize model
    encoder = ENCODER(*ENCODER_ARGS).to(device)
    decoder = DECODER(*DECODER_ARGS).to(device)
    model = nn.ModuleList([encoder, decoder])

    # Initialize Weights and Biases
    name = f"Mode_{MODE}_Encoder_{ENCODER_NAME}_Decoder_{DECODER_NAME}_{i}"
    wandb.init(
        project = "BGLC-EuroSAT-PS-SSL",
        name = name,
        config = hyperparameters,
    )
    save_path = os.path.join("params", name + ".pt")

    # Initialize loss, optimizer, trainer, and early stopper
    loss_func = MixedAELoss(ssim_weight=hyperparameters["ssim_weight"])
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    trainer = SelfSupervisedTrainingModule(encoder, decoder, loss_func, optimizer)
    early_stopper = EarlyStopper(patience=MAX_EPOCHS, maximize=False)

    # Training/validation loop
    for _ in range(MAX_EPOCHS):
        # Training
        trainer.training_epoch(dl_train)
        # Validation
        x_hat_train, x_train, _, _ = trainer.predict(dl_train)
        train_loss = loss_func(x_hat_train, x_train)
        x_hat_val, x_val, _, _ = trainer.predict(dl_val)
        val_loss = loss_func(x_hat_val, x_val)
        # Log state and stop if converged
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        if early_stopper.early_stop(val_loss, checkpoint={"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, save_path=save_path):
            break

    # Reload best performing model
    checkpoint = torch.load(save_path)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    # Compute metrics and example results on test set
    x_hat_test, x_test, y_test, z_test = trainer.predict(dl_test)
    examples = [wandb.Image(norm_for_visualization(torch.cat([x_test[ii], x_hat_test[ii]], dim=1))[[3,2,1]].permute((1,2,0)).numpy()) for ii in range(5)]
    test_l1 = F.l1_loss(x_hat_test, x_test)
    test_ssim = structural_similarity_index_measure(x_hat_test, x_test)

    # Train SVM on z and compute its accuracy
    _, _, y_train, z_train = trainer.predict(dl_train)
    y_train, z_train, y_test, z_test = y_train.squeeze(), z_train.squeeze(), y_test.squeeze(), z_test.squeeze()
    scaler = MinMaxScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)
    classifier = SVC()
    classifier.fit(z_train, y_train)
    y_hat = classifier.predict(z_test)
    acc_svm = accuracy_score(y_test, y_hat)

    # Log results
    wandb.log({
        "examples": examples,
        "test_l1": test_l1,
        "test_ssim": test_ssim,
        "test_acc_svm": acc_svm
    })
    wandb.finish()