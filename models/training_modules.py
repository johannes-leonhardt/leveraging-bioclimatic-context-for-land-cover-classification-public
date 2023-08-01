#%% Imports

import torch

#%% Class for supervised training and inference

class SupervisedTrainingModule:

    def __init__(self, encoder, classifier, loss_func=None, optimizer=None):
        
        self.device = next(encoder.parameters()).device
        
        self.encoder = encoder
        self.classifier = classifier
        self.loss_func = loss_func
        self.optimizer = optimizer

    def training_epoch(self, dl):
        
        self.encoder.train()
        self.classifier.train()

        for _, batch in enumerate(dl):

            x, y, aux = batch
            x = x.to(self.device)
            y = y.to(self.device)
            aux = aux.to(self.device)

            z = self.encoder(x, aux)
            y_hat = self.classifier(z)

            loss = self.loss_func(y_hat, y)

            self.encoder.zero_grad()
            self.classifier.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, dl):

        self.encoder.eval()
        self.classifier.eval()

        y_hat_all, y_all = [], []

        for _, batch in enumerate(dl):

            x, y, aux = batch
            x = x.to(self.device)
            y = y.to(self.device)
            aux = aux.to(self.device)

            z = self.encoder(x, aux)
            y_hat = self.classifier(z)
        
            y_hat_all.append(y_hat.detach().cpu())
            y_all.append(y.cpu())
        
        y_hat_all = torch.concat(y_hat_all, dim=0)
        y_all = torch.concat(y_all, dim=0)

        return y_hat_all, y_all

#%% Class for self-supervised autoencoder training and inference

class SelfSupervisedTrainingModule:

    def __init__(self, encoder, decoder, loss_func=None, optimizer=None):

        self.device = next(encoder.parameters()).device
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.optimizer = optimizer

    def training_epoch(self, dl):

        self.encoder.train()
        self.decoder.train()

        for _, batch in enumerate(dl):

            x, _, aux = batch
            x = x.to(self.device)
            aux = aux.to(self.device)

            z = self.encoder(x, aux)
            x_hat = self.decoder(z, aux)

            loss = self.loss_func(x_hat, x)

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, dl):

        self.encoder.eval()
        self.decoder.eval()
        
        x_hat_all, x_all, y_all, z_all = [], [], [], []

        for _, batch in enumerate(dl):

            x, y, aux = batch
            x = x.to(self.device)
            aux = aux.to(self.device)

            z = self.encoder(x, aux)
            x_hat = self.decoder(z, aux)

            x_hat_all.append(x_hat.detach().cpu())
            x_all.append(x.cpu())
            y_all.append(y)
            z_all.append(z.detach().cpu())

        x_hat_all = torch.concat(x_hat_all, dim=0)
        x_all = torch.concat(x_all, dim=0)
        y_all = torch.concat(y_all, dim=0)
        z_all = torch.concat(z_all, dim=0)

        return x_hat_all, x_all, y_all, z_all