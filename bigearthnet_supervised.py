#%% Imports

import os

import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.classification import multilabel_accuracy, multilabel_f1_score
import wandb

from models.resnet18_encoders import *
from models.classifier import Classifier
from models.training_modules import SupervisedTrainingModule
from models.utils import EarlyStopper

from data.bigearthnet import BigEarthNet, BigEarthNetLoader

#%% Settings and Hyperparameters

# General settings
N_CHANNELS = 12
N_CLASSES = 19
N_AUX = 20
MAX_EPOCHS = 100

# Hyperparameters
hyperparameters = {
    "batch_size": 512,
    "learning_rate": 1e-5
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#%% Datasets and -loaders

df = gpd.read_file(os.path.join('data', 'bigearthnet.gpkg'))

# Training, validation, and testing datasets
ds_train = BigEarthNet(df[df.split == 'train'])
ds_val = BigEarthNet(df[df.split == 'val'])
ds_test = BigEarthNet(df[df.split == 'test'], augmentation=False)

print(f"#Samples: Training: {len(ds_train)}, Validation: {len(ds_val)}, Testing: {len(ds_test)}.")

# Dataloaders
dl_train = BigEarthNetLoader(ds_train, batch_size=hyperparameters["batch_size"])
dl_val = BigEarthNetLoader(ds_val, batch_size=hyperparameters["batch_size"])
dl_test = BigEarthNetLoader(ds_test, batch_size=hyperparameters["batch_size"])

#%% Model settings

# Encoder
ENCODER = ResNet18Encoder
ENCODER_ARGS = [N_CHANNELS, 3]
NAME = "ResNet18"

# Classifier
CLASSIFIER = Classifier
CLASSIFIER_ARGS = [512, N_CLASSES] # First argument needs to dim of latent space

# Path of initial encoder weights from pre-training. Set to None for random initialization.
ENCODER_PATH = None
MODE = "Supervised" if ENCODER_PATH is None else "Downstream"

#%% Training run and quick evaluation

# Initialize model
encoder = ENCODER(*ENCODER_ARGS).to(device)
if MODE == "Downstream":
    checkpoint = torch.load(ENCODER_PATH)
    encoder.load_state_dict(checkpoint["encoder"])
classifier = CLASSIFIER(*CLASSIFIER_ARGS).to(device)
model = nn.ModuleList([encoder, classifier])

# Initialize Weights and Biases
name = f"BEN_Mode_{MODE}_Model_{NAME}"
wandb.init(
    project = "BGLC-BigEarthNet-PS",
    name = name,
    config = hyperparameters,
)
save_path = os.path.join("params", name + ".pt")

# Initialize loss, optimizer, trainer, and early stopper
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
trainer = SupervisedTrainingModule(encoder, classifier, loss_func, optimizer)
early_stopper = EarlyStopper(patience=5, maximize=True)

# Training/validation loop
for _ in range(MAX_EPOCHS):
    # Training
    trainer.training_epoch(dl_train)
    # Validation
    y_hat, y = trainer.predict(dl_train)
    train_acc = multilabel_accuracy(y_hat, y, num_labels=N_CLASSES)
    y_hat, y = trainer.predict(dl_val)
    val_acc = multilabel_accuracy(y_hat, y, num_labels=N_CLASSES)
    wandb.log({"train_acc":train_acc, "val_acc":val_acc})
    if early_stopper.early_stop(val_acc, checkpoint={"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}, save_path=save_path):
        break

# Reload best performing model
checkpoint = torch.load(save_path)
encoder.load_state_dict(checkpoint["encoder"])
classifier.load_state_dict(checkpoint["classifier"])

# Compute metrics on test set
y_hat, y = trainer.predict(dl_test)
test_acc = multilabel_accuracy(y_hat, y, num_labels=N_CLASSES)
test_f1 = multilabel_f1_score(y_hat, y, num_labels=N_CLASSES, average="micro")
test_f1_mac = multilabel_f1_score(y_hat, y, num_labels=N_CLASSES, average="macro")

# Log results
wandb.log({
    "test_acc": test_acc,
    "test_f1": test_f1,
    "test_f1_mac": test_f1_mac
})
wandb.finish()