import torch
from torch.nn.functional import interpolate
import numpy as np


def stack_slice_selection(x,normals,centers = None,D = None,
                        slice_size = 60, interp = True,interp_size = [60,120,120],
                        Volumes_realsize = None, n_slices = 9,
                        voxels_size = None):

    volumes = torch.unsqueeze(x,1)


    if interp:
      volumes = interpolate(volumes,size = interp_size,mode = 'trilinear',align_corners=True)

    min_dist = slice_size // 2

    #linspaces for plane grid
    P = torch.linspace(-0.75,0.75,min_dist*2)
    Q = torch.linspace(-0.75,0.75,min_dist*2)

    N_max = np.amax(np.abs(normals.numpy()),axis=1)

    if N_max == 0:
        # Create Y and Z grids
        YQN,ZQN = torch.meshgrid((P,Q))
        YQN = YQN.repeat([x.shape[0],n_slices,1,1])
        ZQN = ZQN.repeat([x.shape[0],n_slices,1,1])

        VI = torch.zeros_like(YQN)
        XQN = torch.zeros_like(YQN)

    elif N_max ==  1:
        # Create Z and X grids
        ZQN,XQN = torch.meshgrid((P,Q))
        XQN = XQN.repeat([x.shape[0],n_slices,1,1])
        ZQN = ZQN.repeat([x.shape[0],n_slices,1,1])

        VI = torch.zeros_like(XQN)
        YQN = torch.zeros_like(XQN)
    else:
        # Create X and Y grids
        XQN,YQN = torch.meshgrid((P,Q))
        XQN = XQN.repeat([x.shape[0],n_slices,1,1])
        YQN = YQN.repeat([x.shape[0],n_slices,1,1])

        VI = torch.zeros_like(XQN)
        ZQN = torch.zeros_like(XQN)



    total_distance = normals[:,0] * Volumes_realsize[:,0] + normals[:,1] * Volumes_realsize[:,1] + normals[:,2] * Volumes_realsize[:,2]
    slice_distance = normals[:,0] * voxels_size[:,0] + normals[:,1] * voxels_size[:,1] + normals[:,2] * voxels_size[:,2]
    slice_D = slice_distance / (total_distance)

    new_D = False
    D_real = None
    if D is None:
        new_D = True
        # Calculate D from plane equation (Axo + By0 + Cz0)
        D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)) )
        #D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)*Volumes_realsize[:,1]/2) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)*Volumes_realsize[:,0]/2) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)*Volumes_realsize[:,2]/2) )
        D_real = ( torch.mul(normals[:,1],centers[:,1]*Volumes_realsize[:,1]) +  torch.mul(normals[:,0],centers[:,0]*Volumes_realsize[:,0]) + torch.mul(normals[:,2],centers[:,2]*Volumes_realsize[:,2]) )

    normal_eps = torch.zeros_like(normals)
    normal_eps[:,1] = 0.00001
    normals[:,1] = torch.tensor(np.maximum(normals[:,1].numpy(),normal_eps[:,1].numpy()))
    for i,s in enumerate(range(- (n_slices//2), (n_slices//2) + 1)):
        # Calculate Y values from plane equation
        D_n = D + (s*slice_D)

        if N_max == 0:
            XQN[:,i,:,:] = (D_n.view(-1,1,1) - normals[:,1].view(-1,1,1)*YQN[:,i,:,:] - normals[:,2].view(-1,1,1)*ZQN[:,i,:,:]) / normals[:,0].view(-1,1,1)
        elif N_max == 1:
            YQN[:,i,:,:] = (D_n.view(-1,1,1) - normals[:,0].view(-1,1,1)*XQN[:,i,:,:] - normals[:,2].view(-1,1,1)*ZQN[:,i,:,:]) / normals[:,1].view(-1,1,1)
        else:
            ZQN[:,i,:,:] = (D_n.view(-1,1,1) - normals[:,0].view(-1,1,1)*XQN[:,i,:,:] - normals[:,1].view(-1,1,1)*YQN[:,i,:,:]) / normals[:,2].view(-1,1,1)


        # Stack grid values
        grid = torch.unsqueeze(torch.stack((ZQN[:,i,:,:], YQN[:,i,:,:],XQN[:,i,:,:]), 3),1)

        # sample grid values from volumes
        VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))


    if interp:
      volumes = torch.squeeze(volumes)
      # If batch size is 1
      if volumes.dim()<3:
        volumes = torch.unsqueeze(volumes,0)
      if Volumes_realsize is not None:
        XQN = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN)
        YQN = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN)
        ZQN = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN)
        if new_D:
          if D.dim()<2:
            D = torch.unsqueeze(D,0)
            if D_real is not None:
              D_real = torch.unsqueeze(D_real,0)
          return volumes, XQN, YQN, ZQN, VI,D,D_real
        else:
          if D.dim()<2:
            D = torch.unsqueeze(D,0)
            if D_real is not None:
              D_real = torch.unsqueeze(D_real,0)
          return volumes, XQN, YQN, ZQN, VI
    else:
      if Volumes_realsize is not None:
        XQN = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN)
        YQN = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN)
        ZQN = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN)
        if new_D:
          if D.dim()<2:
            D = torch.unsqueeze(D,0)
            if D_real is not None:
              D_real = torch.unsqueeze(D_real,0)
          return XQN, YQN, ZQN, VI,D,D_real
        else:
          if D.dim()<2:
            D = torch.unsqueeze(D,0)
            if D_real is not None:
              D_real = torch.unsqueeze(D_real,0)
          return XQN, YQN, ZQN, VI
