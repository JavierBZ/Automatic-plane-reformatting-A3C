import torch
import numpy as np

# def calcDistParams(params1,params2,scale_d=1e3):
#   dist_ang = params1[0:3] - params2[0:3]
#   dist_ang = torch.sum(torch.abs(dist_ang))
#   dist = (params1[3] - params2[3])*scale_d
#   dist = torch.sum(torch.abs(dist))
#
#
#   return (dist_ang + dist).numpy()

def calcDistParams_np(params1,params2,scale_d=1e3):
  dist_ang = params1[0:3] - params2[0:3]
  dist_ang = np.sum(np.abs(dist_ang))
  return (dist_ang + params2[3])

def calcDistParams2_np(params1,params2,scale_d=1e3):
  dist_ang = params1[0:3] - params2[0:3]
  dist_ang = np.sum(np.abs(dist_ang))
  dist = params2[3]

  return dist, dist_ang

def calcDistParams(params1,params2,scale_d=1e3):
  dist_ang = params1[0:3] - params2[0:3]
  dist_ang = torch.sum(torch.abs(dist_ang))
  dist = (params1[3] - params2[3])*scale_d
  dist = torch.sum(torch.abs(dist))
  #return (dist_ang + params2[3]).numpy()
  return (dist_ang + dist).numpy()

def origins_distance(origin1,origin2):
    #return np.linalg.norm((origin1 - origin2).numpy())
    return torch.cdist(origin1.unsqueeze(0),origin2.unsqueeze(0)).item()

def calcDistParams2(params1,params2,scale_d=1e3):
  dist_ang = params1[0:3] - params2[0:3]
  dist_ang = torch.sum(torch.abs(dist_ang))
  #dist1 = (params1[4] - params2[4])*scale_d
  dist = (params1[3] - params2[3])*scale_d
  dist = torch.sum(torch.abs(dist))
  #dist = params2[3]

  return dist.numpy(), dist_ang.numpy()

def distance_move(translation_m,normal,Volume_realsize):

  total_move = normal[0] * Volume_realsize[0] + normal[1] * Volume_realsize[1] + normal[2] * Volume_realsize[2]
  normalized_translation = translation_m / total_move
  #normalized2_translations  = normalized_translations*2 - 1
  #print(translations_m,total_move,normalized_translations,normalized2_translations)
  return normalized_translation

def map_to_uint8(vi,velocities=False,vel_interp=False):
    # if velocities and not vel_interp:
    #     vi_ = torch.zeros_like(vi,dtype=torch.uint8)
    #     #for ch in range(4):
    #     for ch in range(3):
    #         in_max = torch.max(torch.max(torch.squeeze(vi[ch,:,:,:]),dim=1)[0],dim=1,keepdims=True)[0]
    #         in_min = torch.min(torch.min(torch.squeeze(vi[ch,:,:,:]),dim=1)[0],dim=1,keepdims=True)[0]
    #         vi_[ch,:,:,:] = ( (vi[ch,:,:,:] - in_min.view(-1,1,1)) * ( (255.0 - 0.0) / ( in_max - in_min ) ).view(-1,1,1) + 0.0 ).to(torch.uint8)
    #     return vi_
    # else:
    #     in_max = torch.max(torch.max(vi,dim=1)[0],dim=1,keepdims=True)[0]
    #     in_min = torch.min(torch.min(vi,dim=1)[0],dim=1,keepdims=True)[0]
    #     return ( (vi - in_min.view(-1,1,1)) * ( (255.0 - 0.0) / ( in_max - in_min ) ).view(-1,1,1) + 0.0 ).to(torch.uint8)

    return vi

def checkBackgroundRatio(plane, min_pixel_val=0.5, ratio=0.9):
    ''' check ratio full of background is not larger than (default 80%)
        Returns
            Boolean: true if background pixels
    '''
    total = plane.shape[0] * plane.shape[1]
    # count non-zero pixels larger than (> 0.5)
    nonzero_count = np.count_nonzero((plane.numpy() > min_pixel_val)*1)
    zero_ratio = (total-nonzero_count)/total

    return (zero_ratio > ratio)

def mutual_information(hgram):
     # Mutual information for joint histogram
     # Convert bins counts to probability values
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def Rot3D_matrix(axis,angle):
    if axis == 0:
        M = np.array([[1, 0, 0],[0, np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],dtype=np.float32)
    elif axis == 1:
        #M = np.array([[np.cos(angle), 0, -np.sin(angle)],[0, 1, 0],[np.sin(angle), 0, np.cos(angle)]],dtype=np.float32) #Incorrect
        M = np.array([[np.cos(angle), 0, np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]],dtype=np.float32) # Correct
    elif axis == 2:
        M = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle),0],[0, 0, 1]],dtype=np.float32)

    return torch.from_numpy(M)

def Rot3D_matrix_np(axis,angle):
    if axis == 0:
        M = np.array([[1, 0, 0],[0, np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],dtype=np.float32)
    elif axis == 1:
        M = np.array([[np.cos(angle), 0, -np.sin(angle)],[0, 1, 0],[np.sin(angle), 0, np.cos(angle)]],dtype=np.float32)
    elif  axis == 2:
        M = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle),0],[0, 0, 1]],dtype=np.float32)

    return M

def coors_distance(XQN,YQN,ZQN,XQN2,YQN2,ZQN2):
    Coors =torch.cat((XQN.flatten().unsqueeze(0),YQN.flatten().unsqueeze(0),ZQN.flatten().unsqueeze(0)),0).transpose(1,0)
    Coors2 =torch.cat((XQN2.flatten().unsqueeze(0),YQN2.flatten().unsqueeze(0),ZQN2.flatten().unsqueeze(0)),0).transpose(1,0)

    return torch.mean(torch.cdist(Coors,Coors2)).item()*1e3

class StatCounter(object):
    """ A simple counter"""

    def __init__(self):
        self.reset()

    def feed(self, v):
        """
        Args:
            v(float or np.ndarray): has to be the same shape between calls.
        """
        self._values.append(v)

    def reset(self):
        self._values = []


    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        assert len(self._values)
        return np.mean(self._values)

    @property
    def sum(self):
        assert len(self._values)
        return np.sum(self._values)

    @property
    def max(self):
        assert len(self._values)
        return max(self._values)

    @property
    def min(self):
        assert len(self._values)
        return min(self._values)
    def samples(self):
        """
        Returns all samples.
        """
        return self._values
