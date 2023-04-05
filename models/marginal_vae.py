import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, n_layers, n_params, latent_dim, data_size):
        super(Encoder,self).__init__()    

        self._latent_dim = latent_dim

        self._fc_x = nn.Linear(data_size, n_params)
        self._fc_y = nn.Linear(data_size, n_params)
        layers = []
        layers.append(nn.ReLU())   
        layers.append(nn.Linear(n_params*2,n_params))
        layers.append(nn.ReLU())
        for n in range(n_layers):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_params,2*latent_dim))
        self._layers =  nn.Sequential(*layers)
        self._relu = nn.ReLU()

    def forward(self,x):
        x,y = x[:,0,:],x[:,1,:]
        x_in = self._fc_x(x)
        y_in = self._fc_y(y)
        x = torch.cat((x_in,y_in),1)
        x = self._layers(x)
        mu,sigma = x[:,:self._latent_dim],x[:,self._latent_dim:]
        return mu, sigma
    
    
class Decoder(nn.Module):
    def __init__(self, n_layers, n_params, latent_dim, data_size):
        super(Decoder,self).__init__()    
        self._output_size  = data_size
        layers = []
        layers.append(nn.Linear(latent_dim, n_params))
        layers.append(nn.ReLU())
        for n in range(n_layers):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(nn.ReLU())
        self._layers = nn.Sequential(*layers)

        self._fc_x = nn.Linear(n_params , data_size)   
        self._fc_y = nn.Linear(n_params , data_size)

    def forward(self,z):
        x = self._layers(z)
        x_out = self._fc_x(x)
        y_out = self._fc_y(x)
        return  torch.reshape(torch.cat((x_out,y_out),1),(-1,2,self._output_size))
    
    
    
class VAE(nn.Module):
    def __init__(self,n_layers_e, n_layers_d , n_params, latent_dim, data_size):
        super().__init__()
        self._encoder = Encoder(n_layers_e, n_params, latent_dim, data_size)
        self._decoder = Decoder(n_layers_d, n_params, latent_dim, data_size)
        self._latent_dim = latent_dim
        self._output_size = data_size

    def sample(self,mu,sigma):
        div = torch.device('cpu') if mu.get_device() else torch.device('cuda:0')
        Normal_distrib = torch.distributions.Normal(0, 1)
        Normal_distrib.loc = Normal_distrib.loc.to(div) 
        Normal_distrib.scale = Normal_distrib.scale.to(div)
        return mu + sigma*Normal_distrib.sample(mu.shape)

    def forward(self,x):
        mu,sigma = self._encoder(x)
        latent = self.sample(mu,sigma)
        reconstruction = self._decoder(latent)
        return mu, sigma, reconstruction