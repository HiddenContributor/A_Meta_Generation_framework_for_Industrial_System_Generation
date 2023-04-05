import torch
import torch.nn as nn

system_dim = 180
h_dim = 3
meta_latent_dim = 4
n_params = 64
device = torch.device('cpu')





class Vanilla_Generator(nn.Module):
    def __init__(self,meta_latent_dim):
        super(Vanilla_Generator,self).__init__()
        self.latent_dim = meta_latent_dim
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

class Vanilla_Discriminator(nn.Module):
    def __init__(self,system_dim,h_dim):
        super(Vanilla_Discriminator,self).__init__()
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
        layers.append(nn.Linear(n_params, 1))
        self._layers =  nn.Sequential(*layers)
        self._sig = nn.Sigmoid()
        
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

        x = self._sig(self._layers(x))
        return x


class Vanilla_GAN(nn.Module):
    ## Initialize. Define latent dim, learning rate, and Adam betas 
    def __init__(self,):
        super(Vanilla_GAN,self).__init__()
        self.latent_dim = latent_dim        
        self.gen = Vanilla_Generator(latent_dim,system_dim=system_dim,h_dim=h_dim,n_params=n_params)
        self.disc = Vanilla_Discriminator(system_dim,h_dim,meta_latent_dim=latent_dim,n_params=n_params)
    def forward(self, z,h_param):
        return self.gen(z,h_param)