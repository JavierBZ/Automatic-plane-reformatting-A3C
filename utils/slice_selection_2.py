import torch
from torch.nn.functional import interpolate
import numpy as np
import random


def stack_slice_selection(x,normals,centers = None,D = None,
                        slice_size = 60, interp = True,interp_size = [60,120,120],
                        Volumes_realsize = None, n_slices = 9,
                        voxels_size = None):

    volumes = torch.unsqueeze(x,1)



    if interp:
      volumes = interpolate(volumes,size = interp_size,mode = 'trilinear',align_corners=True)

    x1 = 0.0
    y1 = 0.0
    z1 = 1.0

    if torch.cdist(normals,torch.as_tensor([x1,y1,z1]).float().unsqueeze(0)).item() < 1e-4:
        x1 = 0.0
        y1 = 1.0
        z1 = 0.0
    # x1 = random.random()
    # y1 = random.random()
    # z1 = random.random()

    w1 = torch.unsqueeze(torch.tensor([x1,y1,z1]) / np.linalg.norm(np.array([x1,y1,z1])),dim=0)
    w1 = w1.repeat([normals.shape[0],1])

    w1 = w1 - torch.sum(torch.mul(normals,w1),1).view(volumes.shape[0],1) * normals       # make it orthogonal to k
    w1 = w1/torch.tensor(np.linalg.norm(w1,axis = 1)).view(normals.shape[0],1)
    w2 = torch.cross(w1, normals,dim = 1)

    min_dist = slice_size // 2

    #linspaces for plane grid
    P = torch.linspace(-0.8,0.8,min_dist*2)
    Q = torch.linspace(-0.8,0.8,min_dist*2)

    P,Q = torch.meshgrid((P,Q))

    XQN = P.repeat([x.shape[0],n_slices,1,1])
    XQN = torch.zeros_like(XQN)
    YQN = torch.zeros_like(XQN)
    ZQN = torch.zeros_like(XQN)
    VI = torch.zeros_like(XQN)

    P = P.repeat([x.shape[0],1,1])
    Q = Q.repeat([x.shape[0],1,1])

    total_distance = normals[:,0] * Volumes_realsize[:,0]/2 + normals[:,1] * Volumes_realsize[:,1]/2 + normals[:,2] * Volumes_realsize[:,2]/2
    slice_distance = normals[:,0] * voxels_size[:,0] + normals[:,1] * voxels_size[:,1] + normals[:,2] * voxels_size[:,2]
    slice_D = slice_distance / (total_distance)

    new_D = False
    D_real = None
    if D is None:
        new_D = True
        # Calculate D from plane equation (Axo + By0 + Cz0)
        #D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)) )
        D = ( torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
        # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)*Volumes_realsize[:,1]/2) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)*Volumes_realsize[:,0]/2) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)*Volumes_realsize[:,2]/2) )
        # D_real = ( torch.mul(normals[:,1],centers[:,1]*Volumes_realsize[:,1]) +  torch.mul(normals[:,0],centers[:,0]*Volumes_realsize[:,0]) + torch.mul(normals[:,2],centers[:,2]*Volumes_realsize[:,2]) )

    for i,s in enumerate(range(- (n_slices//2), (n_slices//2) + 1)):
        # Calculate Y values from plane equation
        #D_n = D + (s*slice_D)

        #new_origin = torch.zeros_like(normals) + D_n * normals
        new_origin = centers + (s*slice_D) * normals

        XQN[:,i,:,:] = (new_origin[:,0]).view(-1,1,1) + w1[:,0].view(-1,1,1)*P + w2[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
        YQN[:,i,:,:] = (new_origin[:,1]).view(-1,1,1) + w1[:,1].view(-1,1,1)*P + w2[:,1].view(-1,1,1)*Q # using the two vectors in w
        ZQN[:,i,:,:] = (new_origin[:,2]).view(-1,1,1) + w1[:,2].view(-1,1,1)*P + w2[:,2].view(-1,1,1)*Q

        # Stack grid values
        grid = torch.unsqueeze(torch.stack((ZQN[:,i,:,:], YQN[:,i,:,:],XQN[:,i,:,:]), 3),1)

        # sample grid values from volumes
        VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

    #origin = (torch.zeros_like(normals) + (D * normals * (Volumes_realsize/2))).squeeze()
    origin = (centers * (Volumes_realsize/2)).squeeze()
    #D = ( torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
    D_real = torch.dot(normals.squeeze(),(D * normals * (Volumes_realsize/2)).squeeze())

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
          # if D.dim()<2:
          #   D = torch.unsqueeze(D,0)
          #   if D_real is not None:
          #     D_real = torch.unsqueeze(D_real,0)
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
          # if D.dim()<2:
          #   D = torch.unsqueeze(D,0)
          #   if D_real is not None:
          #     D_real = torch.unsqueeze(D_real,0)
          return XQN, YQN, ZQN, VI,D,D_real,origin
        else:
          if D.dim()<2:
            D = torch.unsqueeze(D,0)
            if D_real is not None:
              D_real = torch.unsqueeze(D_real,0)
          return XQN, YQN, ZQN, VI,origin,D_real
