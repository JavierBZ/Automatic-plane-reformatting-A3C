import numpy as np


def get_affine_matrix(scales,degrees,translation,only_rot=False):


    rads = np.deg2rad(np.array(degrees))
    angle = rads[0]
    M1 = np.array([[1, 0, 0, 0],[0, np.cos(angle),-np.sin(angle), 0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]],dtype=np.float32)
    angle = rads[1]
    M2 = np.array([[np.cos(angle), 0, np.sin(angle),0],[0, 1, 0,0],[-np.sin(angle), 0, np.cos(angle),0],[0,0,0,1]],dtype=np.float32)
    angle = rads[2]
    M3 = np.array([[np.cos(angle), -np.sin(angle), 0, 0],[np.sin(angle), np.cos(angle),0,0],[0, 0, 1,0],[0,0,0,1]],dtype=np.float32)

    M4 = np.matmul(M1,np.matmul(M2,M3))
    # M4 = np.matmul(M1,np.matmul(M3,M2))
    # M4 = np.matmul(M2,np.matmul(M1,M3))
    # M4 = np.matmul(M2,np.matmul(M3,M1))
    # M4 = np.matmul(M3,np.matmul(M1,M2))
    #
    # M4 = np.matmul(M3,np.matmul(M2,M1))



    if only_rot:
        return M4
    else:
        # M4[0,0] = M4[0,0]*scales[0]
        # M4[1,1] = M4[1,1]*scales[1]
        # M4[2,2] = M4[2,2]*scales[2]
        C = 1
        M4[0,3] = translation[0]*C
        M4[1,3] = translation[1]*C
        M4[2,3] = translation[2]*C

        return M4


def rot_3D(degrees,axis=0):


    angle = np.deg2rad(degrees)
    if axis==0:
        M = np.array([[1, 0, 0],[0, np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],dtype=np.float32)
    elif axis==1:
        M = np.array([[np.cos(angle), 0, np.sin(angle)],[0, 1, 0],[-np.sin(angle), 0, np.cos(angle)]],dtype=np.float32)
    else:
        M = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle),0],[0, 0, 1]],dtype=np.float32)

    return M
