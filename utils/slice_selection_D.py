import torch
from torch.nn.functional import interpolate
import numpy as np
import random
from sympy import Plane, Point3D

def stack_slice_selection(x,normals,centers = None,D = None,
                        slice_size = 60, interp = True,interp_size = [60,120,120],
                        Volumes_realsize = None, n_slices = 9,
                        voxels_size = None,mm_size = 150,
                        velocities=False,
                        vel_interp=False,
                        PC_vel=False,
                        orto_2D=False,
                        ori_interp=False,
                        e_V=None):

    if not velocities:
        volumes = torch.unsqueeze(x,1)
    else:
        volumes = x

    total_distance = torch.abs(normals[:,0]) * Volumes_realsize[:,0]/2 + torch.abs(normals[:,1]) * Volumes_realsize[:,1]/2 + torch.abs(normals[:,2]) * Volumes_realsize[:,2]/2
    slice_distance = torch.abs(normals[:,0]) * voxels_size[:,0] + torch.abs(normals[:,1]) * voxels_size[:,1] + torch.abs(normals[:,2]) * voxels_size[:,2]
    slice_D = slice_distance / (total_distance)

    # print(normals,Volumes_realsize,voxels_size)
    # print("SLICE_D", slice_D)
    #slice_D = 0.02

    if orto_2D:
        n_slices = 3


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

    # total_distance = w1[:,0] * Volumes_realsize[:,0]/2 + w1[:,1] * Volumes_realsize[:,1]/2 + w1[:,2] * Volumes_realsize[:,2]/2
    # slice_T1 = (mm_size*1e-3) / (total_distance*2)
    # total_distance = w2[:,0] * Volumes_realsize[:,0]/2 + w2[:,1] * Volumes_realsize[:,1]/2 + w2[:,2] * Volumes_realsize[:,2]/2
    # slice_T2 = (mm_size*1e-3) / (total_distance*2)
    # #linspaces for plane grid
    # P = torch.linspace(-slice_T1.item(),slice_T1.item(),min_dist*2)
    # Q = torch.linspace(-slice_T2.item(),slice_T2.item(),min_dist*2)

    P = torch.linspace(-1,1,min_dist*2)
    Q = torch.linspace(-1,1,min_dist*2)

    P,Q = torch.meshgrid((P,Q))

    XQN = P.repeat([x.shape[0],n_slices,1,1])
    XQN = torch.zeros_like(XQN)
    YQN = torch.zeros_like(XQN)
    ZQN = torch.zeros_like(XQN)
    if velocities:
        VI = torch.zeros((4,XQN.shape[1],XQN.shape[2],XQN.shape[3]))
        #VI = torch.zeros((3,XQN.shape[1],XQN.shape[2],XQN.shape[3]))
    else:
        VI = torch.zeros_like(XQN)

    if PC_vel:
        VI_F = torch.zeros((2,XQN.shape[1],XQN.shape[2],XQN.shape[3]))

    if e_V is not None:
        e_V = e_V.float()
        e_VI = torch.zeros((3,XQN.shape[2],XQN.shape[3],e_V.shape[4]))


    P = P.repeat([x.shape[0],1,1])
    Q = Q.repeat([x.shape[0],1,1])



    new_D = False
    D_real = None
    if D is None:
        new_D = True
        # Calculate D from plane equation (Axo + By0 + Cz0)
        # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)) )
        D = (torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
        #D = torch.dot(normals,centers)
        # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)*Volumes_realsize[:,1]/2) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)*Volumes_realsize[:,0]/2) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)*Volumes_realsize[:,2]/2) )
        # D_real = ( torch.mul(normals[:,1],centers[:,1]*Volumes_realsize[:,1]) +  torch.mul(normals[:,0],centers[:,0]*Volumes_realsize[:,0]) + torch.mul(normals[:,2],centers[:,2]*Volumes_realsize[:,2]) )
    if orto_2D:
        # Calculate Y values from plane equation
        D_n = (torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
        new_origin = D_n * normals
        #new_origin = centers
        XQN[:,0,:,:] = (new_origin[:,0]).view(-1,1,1) + w1[:,0].view(-1,1,1)*P + w2[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
        YQN[:,0,:,:] = (new_origin[:,1]).view(-1,1,1) + w1[:,1].view(-1,1,1)*P + w2[:,1].view(-1,1,1)*Q # using the two vectors in w
        ZQN[:,0,:,:] = (new_origin[:,2]).view(-1,1,1) + w1[:,2].view(-1,1,1)*P + w2[:,2].view(-1,1,1)*Q
        grid = torch.unsqueeze(torch.stack((ZQN[:,0,:,:], YQN[:,0,:,:],XQN[:,0,:,:]), 3),1)
        VI[:,0,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))
        if e_V is not None:
            for pc in range(e_V.shape[4]):
                e_V_pc = e_V[:,:,:,:,pc].squeeze()
                e_VI[:,:,:,pc] = torch.squeeze(torch.nn.functional.grid_sample(e_V_pc.unsqueeze(0),grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

        D_n = (torch.mul(w1[:,1],(centers[:,1])) +  torch.mul(w1[:,0],(centers[:,0])) + torch.mul(w1[:,2],(centers[:,2])) )

        if ori_interp:
            new_origin = D_n * w1
        XQN[:,1,:,:] = (new_origin[:,0]).view(-1,1,1) + normals[:,0].view(-1,1,1)*P + w2[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
        YQN[:,1,:,:] = (new_origin[:,1]).view(-1,1,1) + normals[:,1].view(-1,1,1)*P + w2[:,1].view(-1,1,1)*Q # using the two vectors in w
        ZQN[:,1,:,:] = (new_origin[:,2]).view(-1,1,1) + normals[:,2].view(-1,1,1)*P + w2[:,2].view(-1,1,1)*Q
        grid = torch.unsqueeze(torch.stack((ZQN[:,1,:,:], YQN[:,1,:,:],XQN[:,1,:,:]), 3),1)
        VI[:,1,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

        D_n = (torch.mul(w2[:,1],(centers[:,1])) +  torch.mul(w2[:,0],(centers[:,0])) + torch.mul(w2[:,2],(centers[:,2])) )
        if ori_interp:
            new_origin = D_n * w2
        XQN[:,2,:,:] = (new_origin[:,0]).view(-1,1,1) + w1[:,0].view(-1,1,1)*P + normals[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
        YQN[:,2,:,:] = (new_origin[:,1]).view(-1,1,1) + w1[:,1].view(-1,1,1)*P + normals[:,1].view(-1,1,1)*Q # using the two vectors in w
        ZQN[:,2,:,:] = (new_origin[:,2]).view(-1,1,1) + w1[:,2].view(-1,1,1)*P + normals[:,2].view(-1,1,1)*Q
        grid = torch.unsqueeze(torch.stack((ZQN[:,2,:,:], YQN[:,2,:,:],XQN[:,2,:,:]), 3),1)
        VI[:,2,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

    else:
        for i,s in enumerate(range(- (n_slices//2), (n_slices//2) + 1)):
            # Calculate Y values from plane equation
            D_n = D + (s*slice_D)
            new_origin = D_n * normals
            # new_origin = centers + (s*slice_D) * normals

            XQN[:,i,:,:] = (new_origin[:,0]).view(-1,1,1) + w1[:,0].view(-1,1,1)*P + w2[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
            YQN[:,i,:,:] = (new_origin[:,1]).view(-1,1,1) + w1[:,1].view(-1,1,1)*P + w2[:,1].view(-1,1,1)*Q # using the two vectors in w
            ZQN[:,i,:,:] = (new_origin[:,2]).view(-1,1,1) + w1[:,2].view(-1,1,1)*P + w2[:,2].view(-1,1,1)*Q

            # Stack grid values
            grid = torch.unsqueeze(torch.stack((ZQN[:,i,:,:], YQN[:,i,:,:],XQN[:,i,:,:]), 3),1)

            # sample grid values from volumes
            if velocities:
                VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))
            else:
                VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

    origin = ((D * normals * (Volumes_realsize/2))).squeeze()
    #origin = (centers * (Volumes_realsize/2)).squeeze()
    #D = ( torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
    D_real = torch.dot(normals.squeeze(),(D * normals * (Volumes_realsize/2)).squeeze())

    if vel_interp and not PC_vel:
        VI = (VI[0,:,:,:]*normals.squeeze()[0] + VI[1,:,:,:]*normals.squeeze()[1] + VI[2,:,:,:]*normals.squeeze()[2])/3
    elif PC_vel:
        VI_F[0,:,:,:] = VI[0,:,:,:]
        VI_F[1,:,:,:] = (VI[1,:,:,:]*normals.squeeze()[0] + VI[2,:,:,:]*normals.squeeze()[1] + VI[3,:,:,:]*normals.squeeze()[2])/3
        VI = VI_F

    V_interp = None
    if e_V is not None:
        #e_V = e_V.squeeze()
        V_interp = (e_VI[0,:,:,:]*normals.squeeze()[0] + e_VI[1,:,:,:]*normals.squeeze()[1] + e_VI[2,:,:,:]*normals.squeeze()[2])/3

    if Volumes_realsize is not None:
        XQN = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN)
        YQN = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN)
        ZQN = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN)

        if orto_2D:
            VI = VI.squeeze(0)

        return XQN.squeeze(0), YQN.squeeze(0), ZQN.squeeze(0), VI,D,D_real,origin,V_interp


def line_selection(x,normals,centers = None,D = None,
                        slice_size = 60, interp = True,interp_size = [60,120,120],
                        Volumes_realsize = None, n_slices = 9,
                        voxels_size = None,mm_size = 150,
                        velocities=False,
                        normals2=None,
                        centers2=None,
                        vel_interp=False,
                        switch=False,
                        limits=[0,1],
                        limits_lab=[-1,1],
                        orto_2D=False,
                        PC_vel=False):

    if not velocities:
        volumes = torch.unsqueeze(x,1)
    else:
        volumes = x

    total_distance = torch.abs(normals[:,0]) * Volumes_realsize[:,0]/2 + torch.abs(normals[:,1]) * Volumes_realsize[:,1]/2 + torch.abs(normals[:,2]) * Volumes_realsize[:,2]/2
    slice_distance = torch.abs(normals[:,0]) * voxels_size[:,0] + torch.abs(normals[:,1]) * voxels_size[:,1] + torch.abs(normals[:,2]) * voxels_size[:,2]
    slice_D = slice_distance / (total_distance)

    # print(normals,Volumes_realsize,voxels_size)
    # print("SLICE_D", slice_D)
    #slice_D = 0.02



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

    w1_2 = torch.unsqueeze(torch.tensor([x1,y1,z1]) / np.linalg.norm(np.array([x1,y1,z1])),dim=0)
    w1_2 = w1_2.repeat([normals2.shape[0],1])

    w1_2 = w1_2 - torch.sum(torch.mul(normals2,w1_2),1).view(volumes.shape[0],1) * normals2       # make it orthogonal to k
    w1_2 = w1_2/torch.tensor(np.linalg.norm(w1_2,axis = 1)).view(normals2.shape[0],1)
    w2_2 = torch.cross(w1_2, normals2,dim = 1)

    min_dist = slice_size // 2

    # total_distance = w1[:,0] * Volumes_realsize[:,0]/2 + w1[:,1] * Volumes_realsize[:,1]/2 + w1[:,2] * Volumes_realsize[:,2]/2
    # slice_T1 = (mm_size*1e-3) / (total_distance*2)
    # total_distance = w2[:,0] * Volumes_realsize[:,0]/2 + w2[:,1] * Volumes_realsize[:,1]/2 + w2[:,2] * Volumes_realsize[:,2]/2
    # slice_T2 = (mm_size*1e-3) / (total_distance*2)
    # #linspaces for plane grid
    # P = torch.linspace(-slice_T1.item(),slice_T1.item(),min_dist*2)
    # Q = torch.linspace(-slice_T2.item(),slice_T2.item(),min_dist*2)

    P = torch.linspace(-1,1,min_dist*2)
    Q = torch.linspace(-1,1,min_dist*2)

    P,Q = torch.meshgrid((P,Q))

    P_line = torch.linspace(limits[0],limits[1],501)
    Q_line = torch.linspace(limits[0],limits[1],501)

    P_line,Q_line = torch.meshgrid((P_line,Q_line))

    P_lab = torch.linspace(limits_lab[0],limits_lab[1],501)
    Q_lab = torch.linspace(limits_lab[0],limits_lab[1],501)

    P_lab,Q_lab = torch.meshgrid((P_lab,Q_lab))

    XQN = P.repeat([x.shape[0],n_slices,1,1])
    XQN = torch.zeros_like(XQN)
    YQN = torch.zeros_like(XQN)
    ZQN = torch.zeros_like(XQN)
    if velocities:
        VI = torch.zeros((4,XQN.shape[1],XQN.shape[2],XQN.shape[3]))
        #VI = torch.zeros((3,XQN.shape[1],XQN.shape[2],XQN.shape[3]))
    else:
        VI = torch.zeros_like(XQN)

    XQN_line =  P_line.repeat([x.shape[0],n_slices,1,1])
    YQN_line = torch.zeros_like(XQN_line)
    ZQN_line = torch.zeros_like(XQN_line)
    if velocities:
        VI_line = torch.zeros((4,XQN_line.shape[1],XQN_line.shape[2],XQN_line.shape[3]))
        #VI_line = torch.zeros((3,XQN_line.shape[1],XQN_line.shape[2],XQN_line.shape[3]))
    else:
        VI_line = torch.zeros_like(XQN_line)

    XQN_lab = P_lab.repeat([x.shape[0],n_slices,1,1])
    YQN_lab = torch.zeros_like(XQN_lab)
    ZQN_lab = torch.zeros_like(XQN_lab)
    if velocities:
        VI_lab = torch.zeros((4,XQN_lab.shape[1],XQN_lab.shape[2],XQN_lab.shape[3]))
        #VI_lab = torch.zeros((3,XQN_lab.shape[1],XQN_lab.shape[2],XQN_lab.shape[3]))
    else:
        VI_lab = torch.zeros_like(XQN_lab)

    P = P.repeat([x.shape[0],1,1])
    Q = Q.repeat([x.shape[0],1,1])

    P_line = P_line.repeat([x.shape[0],1,1])
    Q_line = Q_line.repeat([x.shape[0],1,1])

    P_lab = P_lab.repeat([x.shape[0],1,1])
    Q_lab = Q_lab.repeat([x.shape[0],1,1])

    def plane_intersect(a, b):
        """
        a, b   4-tuples/lists
               Ax + By +Cz + D = 0
               A,B,C,D in order

        output: 2 points on line of intersection, np.arrays, shape (3,)
        """
        a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

        aXb_vec = np.cross(a_vec, b_vec)

        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

        # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

        p_inter = np.linalg.solve(A, d).T

        return p_inter[0], (p_inter + aXb_vec)[0],aXb_vec

    new_D = False
    D_real = None
    if D is None:
        new_D = True
        # Calculate D from plane equation (Axo + By0 + Cz0)
        # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)) )
        if switch:
            D = (torch.mul(w2[:,1],(centers[:,1])) +  torch.mul(w2[:,0],(centers[:,0])) + torch.mul(w2[:,2],(centers[:,2])) )
        else:
            D = (torch.mul(w1[:,1],(centers[:,1])) +  torch.mul(w1[:,0],(centers[:,0])) + torch.mul(w1[:,2],(centers[:,2])) )

        D2 = (torch.mul(normals2[:,1],(centers2[:,1])) +  torch.mul(normals2[:,0],(centers2[:,0])) + torch.mul(normals2[:,2],(centers2[:,2])) )

        D_lab =  (torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
        #D = torch.dot(normals,centers)
        # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)*Volumes_realsize[:,1]/2) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)*Volumes_realsize[:,0]/2) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)*Volumes_realsize[:,2]/2) )
        # D_real = ( torch.mul(normals[:,1],centers[:,1]*Volumes_realsize[:,1]) +  torch.mul(normals[:,0],centers[:,0]*Volumes_realsize[:,0]) + torch.mul(normals[:,2],centers[:,2]*Volumes_realsize[:,2]) )


    for i,s in enumerate(range(- (n_slices//2), (n_slices//2) + 1)):
        # Calculate Y values from plane equation

        if orto_2D:
            D_n = D
            D2_n = D2
            D_lab_n = D_lab
        else:
            D_n = D + (s*slice_D)
            D2_n = D2 + (s*slice_D)
            D_lab_n = D_lab + (s*slice_D)

        if switch:
            new_origin = D_n * w2
        else:
            new_origin = D_n * w1
        new_origin2 = D2*normals2
        new_origin_lab = (D_lab_n*normals)

        # print(new_origin,centers2,"idwoqwhdihidqw",new_origin2)
        # a = Plane(Point3D(new_origin[0,0], new_origin[0,1], new_origin[0,2]), normal_vector=(w1[0,0], w1[0,1],w1[0,2]))
        # b = Plane(Point3D(centers2[0,0], centers2[0,1], centers2[0,2]), normal_vector=(normals2[0,0], normals2[0,1],normals2[0,2]))
        #
        # c = a.intersection(b)
        #
        # print(a)
        # print(b)
        # # print(c)
        # # print(c[0])
        # print(c[0].p1.coordinates)
        # print(c[0].p2.coordinates)
        # # exit()

        # new_origin = centers + (s*slice_D) * normals
        if switch:
            A = [w2[0,0],w2[0,1],w2[0,2],D_n[0]]
        else:
            A = [w1[0,0],w1[0,1],w1[0,2],D_n[0]]

        B = [normals2[0,0],normals2[0,1],normals2[0,2],D2_n[0]]

        P_line, N_line, N_vec = plane_intersect(A, B)

        # P_line = np.array(c[0].p2.coordinates,dtype=np.float32)
        # N_line = np.array(c[0].p1.coordinates,dtype=np.float32)

        # N_vec = -np.array(N_line) + np.array(P_line)

        P_line = torch.tensor(-1*N_line).unsqueeze(0)
        #P_line = torch.tensor(N_line).unsqueeze(0)
        N_line = torch.tensor(N_vec).unsqueeze(0)


        if switch:
            XQN[:,i,:,:] = (new_origin[:,0]).view(-1,1,1) + normals[:,0].view(-1,1,1)*P + w1[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
            YQN[:,i,:,:] = (new_origin[:,1]).view(-1,1,1) + normals[:,1].view(-1,1,1)*P + w1[:,1].view(-1,1,1)*Q # using the two vectors in w
            ZQN[:,i,:,:] = (new_origin[:,2]).view(-1,1,1) + normals[:,2].view(-1,1,1)*P + w1[:,2].view(-1,1,1)*Q
        else:
            XQN[:,i,:,:] = (new_origin[:,0]).view(-1,1,1) + normals[:,0].view(-1,1,1)*P + w2[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
            YQN[:,i,:,:] = (new_origin[:,1]).view(-1,1,1) + normals[:,1].view(-1,1,1)*P + w2[:,1].view(-1,1,1)*Q # using the two vectors in w
            ZQN[:,i,:,:] = (new_origin[:,2]).view(-1,1,1) + normals[:,2].view(-1,1,1)*P + w2[:,2].view(-1,1,1)*Q

        XQN_line[:,i,:,:] = (P_line[:,0]).view(-1,1,1) + N_line[:,0].view(-1,1,1)*Q_line #Compute the corresponding cartesian coordinates
        YQN_line[:,i,:,:] = (P_line[:,1]).view(-1,1,1) + N_line[:,1].view(-1,1,1)*Q_line # using the two vectors in w
        ZQN_line[:,i,:,:] = (P_line[:,2]).view(-1,1,1) + N_line[:,2].view(-1,1,1)*Q_line

        if switch:
            XQN_lab[:,i,:,:] = (centers[:,0]).view(-1,1,1) + w1[:,0].view(-1,1,1)*Q_lab #Compute the corresponding cartesian coordinates
            YQN_lab[:,i,:,:] = (centers[:,1]).view(-1,1,1) + w1[:,1].view(-1,1,1)*Q_lab # using the two vectors in w
            ZQN_lab[:,i,:,:] = (centers[:,2]).view(-1,1,1) + w1[:,2].view(-1,1,1)*Q_lab
        else:
            XQN_lab[:,i,:,:] = (centers[:,0]).view(-1,1,1) + w2[:,0].view(-1,1,1)*Q_lab #Compute the corresponding cartesian coordinates
            YQN_lab[:,i,:,:] = (centers[:,1]).view(-1,1,1) + w2[:,1].view(-1,1,1)*Q_lab # using the two vectors in w
            ZQN_lab[:,i,:,:] = (centers[:,2]).view(-1,1,1) + w2[:,2].view(-1,1,1)*Q_lab

        # Stack grid values
        grid = torch.unsqueeze(torch.stack((ZQN[:,i,:,:], YQN[:,i,:,:],XQN[:,i,:,:]), 3),1)

        # sample grid values from volumes
        if velocities:
            VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))
        else:
            VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

        grid = torch.unsqueeze(torch.stack((ZQN_line[:,i,:,:], YQN_line[:,i,:,:],XQN_line[:,i,:,:]), 3),1)

        # sample grid values from volumes
        if velocities:
            VI_line[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))
        else:
            VI_line[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))

        grid = torch.unsqueeze(torch.stack((ZQN_lab[:,i,:,:], YQN_lab[:,i,:,:],XQN_lab[:,i,:,:]), 3),1)

        # sample grid values from volumes
        if velocities:
            VI_lab[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))
        else:
            VI_lab[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))


    origin = ((D * normals * (Volumes_realsize/2))).squeeze()
    #origin = (centers * (Volumes_realsize/2)).squeeze()
    #D = ( torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
    D_real = torch.dot(normals.squeeze(),(D * normals * (Volumes_realsize/2)).squeeze())

    if vel_interp:
        VI = (VI[0,:,:,:]*normals.squeeze()[0] + VI[1,:,:,:]*normals.squeeze()[1] + VI[2,:,:,:]*normals.squeeze()[2])/3
        VI_line = (VI_line[0,:,:,:]*normals.squeeze()[0] + VI_line[1,:,:,:]*normals.squeeze()[1] + VI_line[2,:,:,:]*normals.squeeze()[2])/3
        VI_lab = (VI_lab[0,:,:,:]*normals.squeeze()[0] + VI_lab[1,:,:,:]*normals.squeeze()[1] + VI_lab[2,:,:,:]*normals.squeeze()[2])/3


    if Volumes_realsize is not None:
        XQN = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN)
        YQN = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN)
        ZQN = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN)

        XQN_line = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN_line)
        YQN_line = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN_line)
        ZQN_line = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN_line)

        XQN_lab = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN_lab)
        YQN_lab = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN_lab)
        ZQN_lab = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN_lab)

        return XQN, YQN, ZQN, VI,D,D_real,origin,XQN_line, YQN_line, ZQN_line, VI_line,XQN_lab, YQN_lab, ZQN_lab, VI_lab

#
# def stack_slice_selection(x,normals,centers = None,D = None,
#                         slice_size = 60, interp = True,interp_size = [60,120,120],
#                         Volumes_realsize = None, n_slices = 9,
#                         voxels_size = None):
#
#     volumes = torch.unsqueeze(x,1)
#
#
#
#     if interp:
#       volumes = interpolate(volumes,size = interp_size,mode = 'trilinear',align_corners=True)
#
#     x1 = 0.0
#     y1 = 0.0
#     z1 = 1.0
#
#     if torch.cdist(normals,torch.as_tensor([x1,y1,z1]).float().unsqueeze(0)).item() < 1e-4:
#         x1 = 0.0
#         y1 = 1.0
#         z1 = 0.0
#     # x1 = random.random()
#     # y1 = random.random()
#     # z1 = random.random()
#
#     w1 = torch.unsqueeze(torch.tensor([x1,y1,z1]) / np.linalg.norm(np.array([x1,y1,z1])),dim=0)
#     w1 = w1.repeat([normals.shape[0],1])
#
#     w1 = w1 - torch.sum(torch.mul(normals,w1),1).view(volumes.shape[0],1) * normals       # make it orthogonal to k
#     w1 = w1/torch.tensor(np.linalg.norm(w1,axis = 1)).view(normals.shape[0],1)
#     w2 = torch.cross(w1, normals,dim = 1)
#
#     min_dist = slice_size // 2
#
#     #linspaces for plane grid
#     P = torch.linspace(-0.8,0.8,min_dist*2)
#     Q = torch.linspace(-0.8,0.8,min_dist*2)
#
#     P,Q = torch.meshgrid((P,Q))
#
#     XQN = P.repeat([x.shape[0],n_slices,1,1])
#     XQN = torch.zeros_like(XQN)
#     YQN = torch.zeros_like(XQN)
#     ZQN = torch.zeros_like(XQN)
#     VI = torch.zeros_like(XQN)
#
#     P = P.repeat([x.shape[0],1,1])
#     Q = Q.repeat([x.shape[0],1,1])
#
#     total_distance = normals[:,0] * Volumes_realsize[:,0]/2 + normals[:,1] * Volumes_realsize[:,1]/2 + normals[:,2] * Volumes_realsize[:,2]/2
#     slice_distance = normals[:,0] * voxels_size[:,0] + normals[:,1] * voxels_size[:,1] + normals[:,2] * voxels_size[:,2]
#     slice_D = slice_distance / (total_distance)
#
#     new_D = False
#     D_real = None
#     if D is None:
#         new_D = True
#         # Calculate D from plane equation (Axo + By0 + Cz0)
#         # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)) )
#         D = (torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
#         #D = torch.dot(normals,centers)
#         # D = ( torch.mul(normals[:,1],(centers[:,1]*2 - 1)*Volumes_realsize[:,1]/2) +  torch.mul(normals[:,0],(centers[:,0]*2 - 1)*Volumes_realsize[:,0]/2) + torch.mul(normals[:,2],(centers[:,2]*2 - 1)*Volumes_realsize[:,2]/2) )
#         # D_real = ( torch.mul(normals[:,1],centers[:,1]*Volumes_realsize[:,1]) +  torch.mul(normals[:,0],centers[:,0]*Volumes_realsize[:,0]) + torch.mul(normals[:,2],centers[:,2]*Volumes_realsize[:,2]) )
#
#     for i,s in enumerate(range(- (n_slices//2), (n_slices//2) + 1)):
#         # Calculate Y values from plane equation
#         D_n = D + (s*slice_D)
#         new_origin = D_n * normals
#         # new_origin = centers + (s*slice_D) * normals
#
#         XQN[:,i,:,:] = (new_origin[:,0]).view(-1,1,1) + w1[:,0].view(-1,1,1)*P + w2[:,0].view(-1,1,1)*Q #Compute the corresponding cartesian coordinates
#         YQN[:,i,:,:] = (new_origin[:,1]).view(-1,1,1) + w1[:,1].view(-1,1,1)*P + w2[:,1].view(-1,1,1)*Q # using the two vectors in w
#         ZQN[:,i,:,:] = (new_origin[:,2]).view(-1,1,1) + w1[:,2].view(-1,1,1)*P + w2[:,2].view(-1,1,1)*Q
#
#         # Stack grid values
#         grid = torch.unsqueeze(torch.stack((ZQN[:,i,:,:], YQN[:,i,:,:],XQN[:,i,:,:]), 3),1)
#
#         # sample grid values from volumes
#         VI[:,i,:,:] = torch.squeeze(torch.nn.functional.grid_sample(volumes,grid,align_corners = False,mode='bilinear', padding_mode='zeros'))
#
#     origin = ((D * normals * (Volumes_realsize/2))).squeeze()
#     #origin = (centers * (Volumes_realsize/2)).squeeze()
#     #D = ( torch.mul(normals[:,1],(centers[:,1])) +  torch.mul(normals[:,0],(centers[:,0])) + torch.mul(normals[:,2],(centers[:,2])) )
#     D_real = torch.dot(normals.squeeze(),(D * normals * (Volumes_realsize/2)).squeeze())
#
#
#     if Volumes_realsize is not None:
#         XQN = (Volumes_realsize[:,0]/2).view(-1,1,1,1) * (XQN)
#         YQN = (Volumes_realsize[:,1]/2).view(-1,1,1,1) * (YQN)
#         ZQN = (Volumes_realsize[:,2]/2).view(-1,1,1,1) * (ZQN)
#
#         return XQN, YQN, ZQN, VI,D,D_real,origin
