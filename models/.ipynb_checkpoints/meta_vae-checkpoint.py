import torch
import torch.nn as nn

joint_data_size = 120
density_dim = 30
h_dim= 3
c_size = 120
d_size = 30
device = torch.device('cpu')



class Meta_Encoder(nn.Module):
    def __init__(self,joint_data_size,density_dim,h_dim):
        super(Meta_Encoder,self).__init__()
        self._latent_dim = meta_latent_dim

        self._fch_0 = nn.Linear(h_dim,n_params)
        self._fch_1 = nn.Linear(n_params,n_params)
        self._fch_2 = nn.Linear(n_params,n_params)
        self._fch_3 = nn.Linear(n_params,256)

        self._fcdix = nn.Linear(density_dim,n_params)
        self._fcdiy = nn.Linear(density_dim,n_params)
        self._fcdi_0 = nn.Linear(2*n_params,n_params)
        self._fcdi_1 = nn.Linear(n_params,n_params)
        self._fcdi_2 = nn.Linear(n_params,256)

        self._fcdex = nn.Linear(density_dim,n_params)
        self._fcdey = nn.Linear(density_dim,n_params)
        self._fcde_0 = nn.Linear(2*n_params,n_params)
        self._fcde_1 = nn.Linear(n_params,n_params)
        self._fcde_2 = nn.Linear(n_params,256)

        self._fc_x = nn.Linear(joint_data_size,n_params)
        self._fc_y = nn.Linear(joint_data_size,n_params)
        self._fc_xy = nn.Linear(2*n_params,256)

        self._fc1 = nn.Linear(256*4,n_params)
        self._fc2 = nn.Linear(n_params,n_params)
        self._fc3 = nn.Linear(n_params,n_params)
        self._fc4 = nn.Linear(n_params,n_params)
        self._fc5 = nn.Linear(n_params,2*meta_latent_dim)

        self._relu = nn.ReLU()#nn.ReLU()
        self._sig = nn.Sigmoid()
    def forward(self,x,h,d_int,d_ext):
        
        h = self._relu(self._fch_0(h))
        h = self._relu(self._fch_1(h))
        h = self._relu(self._fch_2(h))
        h = self._relu(self._fch_3(h))

        d_ix,d_iy = d_int[:,0,:],d_int[:,1,:]
        d_ix= self._relu(self._fcdix(d_ix))
        d_iy= self._relu(self._fcdiy(d_iy))
        di = torch.cat((d_ix,d_iy),1)
        di = self._relu(self._fcdi_0(di))
        di = self._relu(self._fcdi_1(di))
        di = self._relu(self._fcdi_2(di))

        d_ex,d_ey = d_ext[:,0,:],d_ext[:,1,:]
        d_ex= self._relu(self._fcdex(d_ex))
        d_ey= self._relu(self._fcdey(d_ey))
        de = torch.cat((d_ex,d_ey),1)
        de = self._relu(self._fcde_0(de))
        de = self._relu(self._fcde_1(de))
        de = self._relu(self._fcde_2(de))

        x,y = x[:,0,:],x[:,1,:]

        x = self._relu(self._fc_x(x))
        y = self._relu(self._fc_y(y))

        x = torch.cat((x,y),1)

        x = self._relu(self._fc_xy(x))
        x = torch.cat((x,h,di,de),1)
        
        x = self._relu(self._fc1(x))
        x = self._relu(self._fc2(x))
        x = self._relu(self._fc3(x))
        x = self._relu(self._fc4(x))
        z = self._fc5(x)
        return z[:,self._latent_dim:],z[:,:self._latent_dim]
    
    
    
class Meta_Decoder(nn.Module):
    def __init__(self,meta_latent_dim):
        super(Meta_Decoder,self).__init__()
        self._cdims = cylinder_latent_dim
        self._latent_dim = meta_latent_dim

        self._fch_0 = nn.Linear(h_dim,n_params)
        self._fch_1 = nn.Linear(n_params,n_params)

        self._fcdi_0 = nn.Linear(n_params,n_params)
        self._fcdi_1 = nn.Linear(n_params,n_params)
        self._fcdi_2 = nn.Linear(n_params,n_params)
        self._fcdi_3 = nn.Linear(n_params,n_params)
        self._fcdi_4 = nn.Linear(n_params,n_params)
        self._fcdi_5 = nn.Linear(n_params,density_latent_dim)

        self._fcde_0 = nn.Linear(n_params,n_params)
        self._fcde_1 = nn.Linear(n_params,n_params)
        self._fcde_2 = nn.Linear(n_params,n_params)
        self._fcde_3 = nn.Linear(n_params,n_params)
        self._fcde_4 = nn.Linear(n_params,n_params)
        self._fcde_5 = nn.Linear(n_params,density_latent_dim)

        self._fc0 = nn.Linear(meta_latent_dim,n_params)
        self._fc1 = nn.Linear(n_params,n_params)
        self._fc2 = nn.Linear(n_params,n_params)
        self._fc3 = nn.Linear(n_params,n_params)
        self._fc4 = nn.Linear(n_params,n_params)
        self._fc5 = nn.Linear(n_params,n_params)
        self._fc6 = nn.Linear(n_params,2*cylinder_latent_dim)

        # self._cat = nn.Linear(4*n_params,n_params)

        self._relu = nn.ReLU()
        self._sig = nn.Sigmoid()
    def forward(self,z,h):
        h = self._relu(self._fch_0(h))
        h = self._relu(self._fch_1(h))

        z = self._relu(self._fc0(z))
        z = z+h

        di = self._relu(self._fcdi_0(z)) 
        di = self._relu(self._fcdi_1(di)) 
        di = self._relu(self._fcdi_2(di)) 


        de = self._relu(self._fcde_0(z)) 
        de = self._relu(self._fcde_1(de)) 
        de = self._relu(self._fcde_2(de))

        x = self._relu(self._fc1(z))
        x = self._relu(self._fc2(x))
        x = self._relu(self._fc3(x))
        x = self._relu(self._fc4(x))

        # cat = torch.cat((di,de,x,h),1)#concat h aussi?
        # cat = self._relu(self._cat(cat))



        # di = di+cat
        di = self._relu(self._fcdi_3(di)) 
        # di = self._relu(self._fcdi_4(di))
        di = self._fcdi_5(di)

        # de = de+cat 
        de = self._relu(self._fcde_3(de)) 
        # de = self._relu(self._fcde_4(de))
        de = self._fcde_5(de)

        
        # x = x+cat
        x = self._relu(self._fc5(x))
        x = self._fc6(x)
        
        return x[:,:self._cdims],x[:,self._cdims:],di,de
    
    
    
class Meta_VAE(nn.Module):
    def __init__(self):
        super(Meta_VAE,self).__init__()
        self.latent_dim=latent_dim
        self._encoder = Meta_Encoder(joint_data_size,density_dim,h_dim,n_params).to(device)
        self._decoder = Meta_Decoder(latent_dim,cylinder_latent_dim ,n_params).to(device)        
    def sample(self,mu,sigma):
        div = torch.device('cpu') if mu.get_device() else torch.device('cuda:0')
        Normal_distrib = torch.distributions.Normal(0, 1)
        Normal_distrib.loc = Normal_distrib.loc.to(div) 
        Normal_distrib.scale = Normal_distrib.scale.to(div)
        return mu + sigma*Normal_distrib.sample(mu.shape)
    def forward(self,data, h_param, d_int,d_ext):
        mu,sigma = self._encoder(data, h_param, d_int,d_ext)
        latent = self.sample(mu,sigma)
        z_int, z_ext, zdi,zde = self._decoder(latent,h_param)
        return mu,sigma, z_int, z_ext, zdi,zde