import torch
import torch.nn as nn

system_dim = 180
h_dim = 3
meta_latent_dim = 4
n_params = 64
device = torch.device('cpu')

class Vanilla_Encoder(nn.Module):
    def __init__(self,system_dim,h_dim,meta_latent_dim,n_params):
        super(Vanilla_Encoder,self).__init__()
        self._latent_dim = meta_latent_dim
        self._relu = nn.ReLU()#nn.ReLU()
        self._fch_0 = nn.Linear(h_dim,n_params)
        self._fch_1 = nn.Linear(n_params,n_params)
        self._fch_2 = nn.Linear(n_params,n_params)
        self._fch_3 = nn.Linear(n_params,256)

        self._fcx = nn.Linear(system_dim,n_params)
        self._fcy = nn.Linear(system_dim,n_params)
        self._fcxy = nn.Linear(2*n_params,256)

        self._fccat = nn.Linear(2*256,n_params)#cat with h

        layers = []       
        for n in range(3):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(self._relu)
        layers.append(nn.Linear(n_params, 2*meta_latent_dim))
        self._layers =  nn.Sequential(*layers)

        
    def forward(self,x,h):
        
        h = self._relu(self._fch_0(h))
        h = self._relu(self._fch_1(h))
        h = self._relu(self._fch_2(h))
        h = self._relu(self._fch_3(h))

        x,y = x[:,0,:],x[:,1,:]
        x = self._relu(self._fcx(x))
        y = self._relu(self._fcy(y))
        x = torch.cat((x,y),1)
        x = self._relu(self._fcxy(x))

        x = torch.cat((x,h),1)
        x = self._relu(self._fccat(x))

        z = self._layers(x)
        return z[:,self._latent_dim:],z[:,:self._latent_dim]
    
class Vanilla_Decoder(nn.Module):
    def __init__(self,meta_latent_dim,system_dim,h_dim,n_params):
        super(Vanilla_Decoder,self).__init__()
        self._latent_dim = meta_latent_dim
        self._sdim = system_dim
        self._relu = nn.ReLU()#nn.ReLU()

        self._fch_0 = nn.Linear(h_dim,n_params)
        self._fch_1 = nn.Linear(n_params,n_params)

        self._fcz = nn.Linear(meta_latent_dim,n_params)

        layers = []      
        for n in range(3):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(self._relu)
        layers.append(nn.Linear(n_params,2*system_dim))
        self._layers =  nn.Sequential(*layers)

    def forward(self,z,h):
        h = self._relu(self._fch_0(h))
        h = self._relu(self._fch_1(h))

        z = self._relu(self._fcz(z))
        z = z+h
        s = self._layers(z)
        s = torch.reshape(s,(-1,2,self._sdim))
        return s
    
    
class Vanilla_VAE(nn.Module):
    def __init__(self,device=device):
        super(Vanilla_VAE,self).__init__()
        self.latent_dim=meta_latent_dim
        self._encoder = Vanilla_Encoder(system_dim,h_dim,meta_latent_dim,n_params).to(device)
        self._decoder = Vanilla_Decoder(meta_latent_dim,system_dim,h_dim,n_params).to(device)        
    def sample(self,mu,sigma):
        div = torch.device('cpu') if mu.get_device() else torch.device('cuda:0')
        Normal_distrib = torch.distributions.Normal(0, 1)
        Normal_distrib.loc = Normal_distrib.loc.to(div) 
        Normal_distrib.scale = Normal_distrib.scale.to(div)
        return mu + sigma*Normal_distrib.sample(mu.shape)
    def forward(self,data, h_param):
        mu,sigma = self._encoder(data, h_param)
        latent = self.sample(mu,sigma)
        recon = self._decoder(latent,h_param)
        return mu,sigma, recon