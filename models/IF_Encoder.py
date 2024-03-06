import torch
import torch.nn as nn

from utils import *
device = get_default_device()

class IF_Encoder(nn.Module):
    def __init__(self ,input_size):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(input_size * 0.2)),
            nn.ReLU(True),
            nn.Linear(int(input_size * 0.2), int(input_size * 0.05)))
        
        self.decoder = nn.Sequential(
            nn.Linear(int(input_size * 0.05), int(input_size * 0.2)),
            nn.ReLU(True),
            nn.Linear(int(input_size * 0.2), input_size), 
            nn.Tanh())
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, n):
        z = self.encoder(batch)
        w = self.decoder(z)
        loss = torch.mean((batch-w)**2)
        return loss

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w = self.decoder(z)
        loss = torch.mean((batch-w)**2)
        return {'validation loss': loss}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['validation loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'validation loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], validation loss: {:.4f}".format(epoch, result['validation loss']))
    
def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam, lr=1e-4):
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()), lr=lr)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
            #Train AE
            loss = model.training_step(batch,epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        del result# history.append(result)
    # return history
    
def testing(model, test_loader, alpha=1.0, beta=0.0):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,'cpu')
        w = model(batch)
        results.append(torch.mean((batch-w)**2,axis=1))
    return results

def encoding(model, data_loader):
    latent=[]
    for [batch] in data_loader:
        batch=to_device(batch,'cpu')
        z = model.encoder(batch)
        latent.append(z)
    return latent