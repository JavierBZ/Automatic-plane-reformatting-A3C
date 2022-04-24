from collections import (Counter, defaultdict, deque, namedtuple)
from collections import deque
import copy

import gym
from gym import spaces

import numpy as np
import csv

import torch

import random

import time

from data_loaders import get_loaders

from utils.env_helper import *
#from utils.slice_selection import stack_slice_selection
from utils.slice_selection_D import stack_slice_selection,line_selection

from skimage.metrics import structural_similarity as ssim

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



class MedicalPlayer(gym.Env):
    """Class that provides 3D medical image environment.
    This is just an implementation of the classic "agent-environment loop".
    Each time-step, the agent chooses an action, and the environment returns
    an observation and a reward."""

    def __init__(self,
                data_loader = None,
                max_num_frames = 500,
                train = True,
                screen_dims = (9,70,70),
                spacing = None,
                history_length=30,
                supervised = False,
                scale_d = 1e3,
                angle_step=5,
                dist_step=5,
                NUMPY = False,
                device = 'cpu',
                eval_velocities=False,
                discrete = False,
                cluster = True,
                isotropic = True,
                Drive = False,
                exp_reward = False,
                done_action = False,
                contrast = 'IPCMRA',
                CD_filt = True,
                Plane = 'PAP',
                velocities = False,
                vel_interp = False,
                PC_vel=False,
                folder=0,
                orto_2D=False,
                only_move=False,
                obs_2=False
                ):


      super(MedicalPlayer, self).__init__()

      # Set data
      train_list1 = ["000a","001a","002a","003a","004a","005a","006a","007a","008a","009a","010a","011a","012b","013a"]

      #train_list2 = ["043","044","045","047","048","049","051","053"]
      train_list2 = ["044","045","047","048","049","051","053"]
      train_list3 = ["083","085","086","091"]
      train_list4 = ["150","151","154","157"]
      #train_list5 = ["200","201","202","203","204","205","210","211","212","215"]
      train_list5 = ["200","201","203","204","205","210","211","212","215"]
      train_list6 = ["221","222","223","224","228","229",
                     "230","231","232","234","236","237","238","239",
                     "241","245","246","247","248",
                     "250"]

      valid_list1 = ["014a","017a"]
      valid_list2 = ["046","052"]
      valid_list3 = ["088"]
      valid_list4 = ["152"]
      valid_list5 = ["202","206","207","213","219"]
      valid_list6 = ["225","226","233","243","244"]

      test_list1 = ["016a","015a"]
      test_list2 = ["054","050"]
      test_list3 = ["087"]
      test_list4 = ["155"]
      test_list5 = ["208","209","218"]
      #test_list5 = ["208","209"]#,"218"]
      test_list6 = ["227","235","240"]
      #test_list7 = ["251","252","253","255","256","257","259"]


      list1 = train_list1 + valid_list1 + test_list1
      list2 = train_list2 + valid_list2 + test_list2
      list3 = train_list3 + valid_list3 + test_list3
      list4 = train_list4 + valid_list4 + test_list4
      list5 = train_list5 + valid_list5 + test_list5
      list6 = train_list6 + valid_list6 + test_list6 + ["251","252","255","256"]


      list2 = ["043","044","047","048","049","051","053","054"]
      list7 = ["300","302","303","304","306","307","308","309"]

      list8 = ["150","152","154","155","163"]

      train_list1, valid_list1  = train_test_split(list1, test_size=0.5, random_state=11)
      train_list2, valid_list2  = train_test_split(list2, test_size=0.5, random_state=11)
      train_list3, valid_list3  = train_test_split(list3, test_size=0.5, random_state=11)
      train_list4, valid_list4  = train_test_split(list4, test_size=0.5, random_state=11)
      train_list5, valid_list5  = train_test_split(list5, test_size=0.5, random_state=11)
      train_list6, valid_list6  = train_test_split(list6, test_size=0.5, random_state=11)
      train_list7, valid_list7  = train_test_split(list7, test_size=0.5, random_state=11)
      #train_list8, valid_list8  = train_test_split(list7, test_size=0.5, random_state=11)


      valid_list1, test_list1  = train_test_split(valid_list1, test_size=0.6, random_state=11)
      valid_list2, test_list2  = train_test_split(valid_list2, test_size=0.6, random_state=11)
      valid_list3, test_list3  = train_test_split(valid_list3, test_size=0.6, random_state=11)
      valid_list4, test_list4  = train_test_split(valid_list4, test_size=0.6, random_state=11)
      valid_list5, test_list5  = train_test_split(valid_list5, test_size=0.6, random_state=11)
      valid_list6, test_list6  = train_test_split(valid_list6, test_size=0.6, random_state=11)
      valid_list7, test_list7  = train_test_split(valid_list7, test_size=0.6, random_state=11)
      #valid_list8, test_list8  = train_test_split(valid_list7, test_size=0.6, random_state=11)

      #test_list5.remove('210')
      #test_list5.remove('203')

      # if Plane == 'PAP' or Plane == 'PAD':
      #      train_list5.remove('219')


      if contrast == 'IPCMRA':
          self.train_list = train_list1 + train_list5 + train_list7
          self.valid_list = valid_list1 + valid_list5 + valid_list7
          self.test_list = test_list1 + test_list5 + test_list7
          #self.test_list.remove("306")
      elif contrast == 'CD':
          self.train_list = train_list2 + train_list6
          self.valid_list = valid_list2 + valid_list6
          self.test_list = test_list2 + test_list6
      else:
          self.train_list = train_list1 + train_list5 + train_list6  + train_list7 + train_list2 + ["150","152"]#+ train_list7 + train_list2  #+ train_list7 + train_list2
          self.valid_list = valid_list1  + valid_list5 + valid_list6 + valid_list7 + valid_list2 + ["155"]
          self.test_list = test_list1 + test_list5 + test_list6 + test_list7 + test_list2 + ["154","163"]


      #self.test_list = ['244'] + test_list7 + test_list2 #test_list1 + test_list5 + test_list6 #
      #self.test_list = test_list1 + test_list5 + test_list6
      #self.test_list = list8


      # Cross Validation
      self.folder = folder
      all_lists  = []

      all_lists.append(list1)
      all_lists.append(list2)
      all_lists.append(list5)
      all_lists.append(list6)
      all_lists.append(list7)
      all_lists.append(list8)

      Y = []
      data_list = []
      for y,l in enumerate(all_lists):
          for p in l:
              Y.append(y)
              data_list.append(p)

      skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=11)
      skf = StratifiedKFold(n_splits=4)

      data_list = np.array(data_list)
      Y = np.array(Y)

      train_lists = []
      valid_lists = []
      test_lists = []
      # print(data_list)
      # print(Y)
      # exit()
      for train_index, test_index in skf.split(data_list,Y):
              # train_index_aux = train_index[:int((len(train_index)*2)/3)]
              # valid_index = train_index[int((len(train_index)*2)/3):]
              train_index_aux, valid_index = train_test_split(train_index,test_size=0.33333, random_state=11,stratify=Y[train_index])

              train_lists.append(list(data_list[train_index_aux]))
              valid_lists.append(list(data_list[valid_index]))
              test_lists.append(list(data_list[test_index]))


      # print('------------------------------------')
      # for l in train_lists:
      #      print(l)
      #      print('LARGO  ',len(l))
      # print('------------------------------------')
      # for l in valid_lists:
      #     print(l)
      #     print('LARGO  ',len(l))
      # print('------------------------------------')
      # for l in test_lists:
      #      print(l)
      #      print('LARGO  ',len(l))


      self.train_list = train_lists[folder]
      self.valid_list = valid_lists[folder]
      self.test_list = test_lists[folder]

      if self.folder == 3:
          self.test_list.remove("163")
          #seslf.test_list.append("053")

      #print(self.test_list)

      #self.test_list = self.train_list + self.valid_list  + self.test_list
      #self.test_list = ["044","201","150"]
      #self.test_list = ["308"]
      #self.test_list = ["219","219"]



      #self.test_list  = self.train_list +  self.valid_list + self.test_list
      # train_list1 = ["Patient_018_phase1","Patient_019_phase1","Patient_021_phase1","Patient_018_phase2",
      #           "Patient_022_phase2","Patient_023_phase1","Patient_023_phase2","Patient_021_phase2"]
      # train_list2 = ["Patient_026_phase1","Patient_026_phase2","Patient_027_phase1","Patient_028_phase1","Patient_029_phase2",
      #               "Patient_030_phase1","Patient_030_phase2", "Patient_031_phase1","Patient_031_phase2","Patient_032_phase1",
      #               "Patient_033_phase1","Patient_035_phase1","Patient_035_phase2","Patient_036_phase1","Patient_036_phase2",
      #               "Patient_029_phase1","Patient_040_phase2","Patient_042_phase2"]
      # train_list3 = ["axial_full_pat00","axial_full_pat01","axial_full_pat03","axial_full_pat06",
      #               "axial_full_pat07","axial_full_pat08","axial_full_pat11","axial_full_pat14",
      #               "axial_full_pat10","axial_full_pat13","axial_full_pat17","axial_full_pat18"]
      #
      # valid_list1 = ["Patient_019_phase2"]
      # valid_list2 = ["Patient_027_phase2","Patient_032_phase2","Patient_040_phase1"]
      # valid_list3 = ["axial_full_pat02","axial_full_pat19"]
      #
      # test_list1 = ["Patient_022_phase1"]
      # test_list2 = ["Patient_028_phase2","Patient_033_phase2","Patient_042_phase1"]
      # test_list3 = ["axial_full_pat12","axial_full_pat05"]
      #
      # self.train_list = train_list1 + train_list2 + train_list3
      # self.valid_list = valid_list1 + valid_list2 + valid_list3
      # self.test_list = test_flist1 + test_list2 + test_list3

      # self.test_list = test_list1 + test_list5
      # #
      # self.test_list.append("051")
      # self.test_list.append("2")
      #self.test_list.remove("047")

      #self.test_list = ["306","250"]

     # self.test_list = ["Patient_033_phase1","axial_full_pat12"]
      #self.test_list = ["244","307"]

      #self.test_list = self.train_list + self.valid_list + self.test_list + list2
      #self.test_list = ["049"]

      #self.test_list = ["302","004a"]
      #self.train_list = self.test_list


      # TL = self.train_list
      # self.train_list = self.valid_list + self.test_list
      # self.valid_list = self.valid_list + self.test_list
      # self.test_list = TL

      # print("TRAIN: ", self.train_list)
      # print("VALID: ",self.valid_list)
      # print("TEST: ",self.test_list)

      self.Plane = Plane
      self.velocities = velocities
      self.vel_interp = vel_interp
      self.PC_vel = PC_vel
      self.only_move = only_move
      self.eval_velocities = eval_velocities

      if cluster and not Drive:
        #if isotropic:
          #data_path = "/home1/jebisbal/Data_isotropic_cluster_eqclahe/"
        data_path = "/home1/jebisbal/Data_def_min_longreg/"

        if Plane == 'PAP':
          labels_path = "/home1/jebisbal/Plane_Labels_longreg/PAP/"
        elif Plane == 'PAD':
            labels_path = "/home1/jebisbal/Plane_Labels_longreg/PAD/"
        elif Plane == 'RPA':
            labels_path = "/home1/jebisbal/Plane_Labels_longreg/RPA/"
        elif Plane == 'LPA':
            labels_path = "/home1/jebisbal/Plane_Labels_longreg/LPA/"
        elif Plane == 'AAsc':
            labels_path = "/home1/jebisbal/Plane_Labels_longreg/AAsc/"

        # data_path = "/home1/jebisbal/Data_def_min_longreg_nacha/"
        # if Plane == 'PAP':
        #   labels_path = "/home1/jebisbal/Plane_Labels_longreg_nacha/PAP/"
        # elif Plane == 'PAD':
        #     labels_path = "/home1/jebisbal/Plane_Labels_longreg_nacha/PAD/"
        # elif Plane == 'RPA':
        #     labels_path = "/home1/jebisbal/Plane_Labels_longreg_nacha/RPA/"
        # elif Plane == 'LPA':
        #     labels_path = "/home1/jebisbal/Plane_Labels_longreg_nacha/LPA/"
        # elif Plane == 'AAsc':
        #     labels_path = "/home1/jebisbal/Plane_Labels_longreg_nacha/AAsc/"

      elif Drive:
        if isotropic:
          data_path = "gdrive/My Drive/11_semestre/Automatic_reformatting/Data_4D_FLOW_2021/Data_isotropic_cluster_rs/"
        else:
          data_path = "/home1/jebisbal/Data_Drive_cluster/"

        if Plane == 'PAP':
            labels_path = "gdrive/My Drive/11_semestre/Automatic_reformatting/Plane_Labels/Labels/PAP/"
        elif Plane == 'PAD':
            labels_path = "gdrive/My Drive/11_semestre/Automatic_reformatting/Plane_Labels/Labels/PAD/"
        elif Plane == 'RPA':
            labels_path = "gdrive/My Drive/11_semestre/Automatic_reformatting/Plane_Labels/Labels/RPA/"
        elif Plane == 'LPA':
            labels_path = "gdrive/My Drive/11_semestre/Automatic_reformatting/Plane_Labels/Labels/LPA/"
        elif Plane == 'AAsc':
            labels_path = "gdrive/My Drive/11_semestre/Automatic_reformatting/Plane_Labels/Labels/AAsc/"

      else:
        if isotropic:
          #data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_longreg/"
          #data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_affinereg/"
          if obs_2:
            #data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_longreg_nacha/"
            data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_longreg/"
          else:
            data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_longreg/"
          #data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_longreg_back_21_12/"
          #data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Data_def_min_normalreg/"
          #data_path = "/Volumes/LAB MICRO/4D_flow_data/Data_def_complete/"

        #data_path = "/Volumes/HV620S/Data_Javier2/Data_Drive/"
        #LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg_labels/"
        #LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg+box_labels_affinereg/"
        if obs_2:
            LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg+box_labels_longreg_nacha/"
        else:
            LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg+box_labels_longreg/"

        #LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg+box_labels_longreg/"
        #LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg+box_labels_longreg_back_21_12/"
        #LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Reg+box_labels_normalreg/"
        #LP = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/"
        if Plane == 'PAP':
          labels_path = LP + "PAP/"
        elif Plane == 'PAD':
            labels_path = LP + "PAD/"
        elif Plane == 'RPA':
            labels_path = LP + "RPA/"
        elif Plane == 'LPA':
            labels_path = LP + "LPA/"
        elif Plane == 'AAsc':
            labels_path = LP + "AAsc/"
        elif Plane == 'Nacha':
            labels_path =  LP +"Nacha/"

      # data_path = "/home1/jebisbal/Datos_para_Javier/"
      # labels_path = "/home1/jebisbal/Datos_para_Javier/"
      #
      # data_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Datos_para_Javier/"
      # labels_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Data/Datos_para_Javier/"

        #labels_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/Labels/"


      if data_loader == 'train':
          self.data_loader, _, _ = get_loaders(self.train_list,self.valid_list,self.test_list,data_path=data_path,labels_path=labels_path,CD_filt=CD_filt,velocities = self.velocities)
      elif data_loader == 'valid':
          _,self.data_loader, _ = get_loaders(self.train_list,self.valid_list,self.test_list,data_path=data_path,labels_path=labels_path,CD_filt=CD_filt,velocities = self.velocities)
      else:
          _,_,self.data_loader = get_loaders(self.train_list,self.valid_list,self.test_list,data_path=data_path,labels_path=labels_path,CD_filt=CD_filt,velocities = self.velocities)

      self.data_iterator = iter(enumerate(self.data_loader))
      self.data_idx = 0


      #######################################################################
      ## save results in csv file
      self.csvfile = 'dummy.csv'
      if not train:
          with open(self.csvfile, 'w') as outcsv:
              fields = ["filename", "dist_error", "angle_error"]
              writer = csv.writer(outcsv)
              writer.writerow(map(lambda x: x, fields))
      #######################################################################
      #self.device = device

      self.orto_2D = orto_2D
      self.epochs = 0
      self.loader_resets = 0
      self.loader_reset_bool = False
      self.NUMPY = NUMPY
      self.reset_stat()
      self._supervised = supervised
      self._init_action_angle_step = angle_step
      self._init_action_dist_step = dist_step
      self.scale_d = scale_d

      # maximum number of frames (steps) per episodes
      self.max_num_frames = max_num_frames
      # stores information: terminal, score, distError
      self.info = None
      # training flag
      self.train = train

      # # image dimension (2D/3D)
      self.screen_dims = screen_dims

      # plane sampling spacings
      self.init_spacing = np.array(spacing)
      # stat counter to store current score or accumlated reward
      self.current_episode_score = StatCounter()
      # get action space and minimal action set
      if discrete:
          if done_action:
              self.action_space = spaces.Discrete(9) # change number actions here
          else:
              self.action_space = spaces.Discrete(8) # change number actions here
      else:
          self.action_space = self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
            )
      if discrete:
          self.actions = self.action_space.n
      else:
          self.actions = 4
      self.observation_space = spaces.Box(low=0, high=1,
                                            shape=self.screen_dims)
      # history buffer for storing last locations to check oscillations
      self._history_length = history_length
      # circular buffer to store plane parameters history [4,history_length]
      self._total_plane_history = deque(maxlen=self.max_num_frames)
      self._plane_history = deque(maxlen=self._history_length)
      self._normal_history = deque(maxlen=self._history_length)
      self._origin_history = deque(maxlen=self._history_length)
      self._bestq_history = deque(maxlen=self._history_length)
      self._dist_history = deque(maxlen=self._history_length)
      self._dist_history_params = deque(maxlen=self._history_length)
      self._dist_supervised_history = deque(maxlen=self._history_length)
      # self._loc_history = [(0,) * self.dims] * self._history_length
      self._loc_history = [(0,) * 4] * self._history_length
      self._qvalues_history = [(0,) * self.actions] * self._history_length
      self._qvalues = [0,] * self.actions

      self.oscillations = 0
      self.n_finish = 0
      self.n_finish_ori = 0
      self.ori_finish = True

      if Plane == 'PAD': #or Plane != 'LPA':
          self.ori_interp = False
      else:
          self.ori_interp = True
      self.ori_interp = False


      self.discrete = discrete
      self.exp_reward = exp_reward
      self.done_action = done_action
      self._restart_episode()

    def reset(self):
        self._restart_episode()

        if self.NUMPY:
          return self.VI.numpy()#, self.loader_reset_bool
        else:
          return self.plane_state#, self.loader_reset_bool
        #return np.expand_dims(np.squeeze(self.VI.numpy()[4,:,:]),0)

    def _reset_loader(self):
        self.data_iterator = iter(enumerate(self.data_loader))
        # print("---------------------")
        # print("Data loader restarted")
        # print("---------------------")
        self.data_idx = 0
        self._restart_episode()

        if self.NUMPY:
          return self.VI.numpy()
        else:
          return self.plane_state


    def _restart_episode(self,new_try=False):
        """
        restart current episoide
        """
        if new_try:
            self.tries += 1
        else:
            self.tries = 0
        self.terminal = False
        self.oscillations = 0
        self.n_finish = 0
        self.n_finish_ori = 0
        self.ori_finish = True
        self.cnt = 0 # counter to limit number of steps per episodes
        self.num_games.feed(1)
        self.current_episode_score.reset()  # reset score stat counter
        self._plane_history.clear()
        self._normal_history.clear()
        self._origin_history.clear()
        self._bestq_history.clear()
        self._dist_history.clear()
        self._dist_history_params.clear()
        self._dist_supervised_history.clear()
        # self._loc_history = [(0,) * self.dims] * self._history_length
        self._loc_history = [(0,) * 4] * self._history_length
        self._qvalues_history = [(0,) * self.actions] * self._history_length

        self.new_random_game(new_try=new_try)

    def new_random_game(self,new_try=False):
        # print('\n============== new game ===============\n')
        self.terminal = False
        self.viewer = None

        if not new_try:
            # sample new volume
            if self.data_idx < len(self.data_loader):
              self.loader_reset_bool = False
              _,data = next(self.data_iterator)
              self.data_idx+=1
            else:
              self.data_iterator = iter(enumerate(self.data_loader))
              # print("---------------------")
              # print("Data loader restarted")
              # print("---------------------")
              self.loader_resets+=1
              self.loader_reset_bool = True
              _,data = next(self.data_iterator)
              self.data_idx = 1
              self.epochs+=1
            self.x = torch.squeeze(data['x'])
            self.y = torch.squeeze(data['y'])
            self.spacing = torch.squeeze(data['voxel_size'])
            self.SEG = torch.squeeze(data['SEG'])
            #self.SEG = None
            self.Patient = data['Patient'][0]
            if self.eval_velocities:
                self.e_V = data['e_V'].squeeze()
            else:
                self.e_V = None

            self.Volume_realsize = self.y[6:]
            #self.Volume_realsize = torch.tensor(self.x.shape,dtype=torch.float) * torch.tensor(self.spacing,dtype=torch.float) * 1e-3

            self.normal_lab = self.y[3:6]

            abs_vector = np.linalg.norm(self.normal_lab)
            if (abs_vector==0):  abs_vector = np.finfo(self.normal_lab.numpy().dtype).eps
            self.normal_lab = self.normal_lab / abs_vector
            self.normal_lab_angles = torch.acos(self.normal_lab)


            self.action_angle_step = copy.deepcopy(self._init_action_angle_step)
            self.action_dist_step = copy.deepcopy(self._init_action_dist_step)

            self._image_dims = self.x.shape[1:]

        # find center point of the initial plane
        if self.train: # +-10% away from center

          if self.only_move:
               x = random.uniform(0.4,0.6)
               y = random.uniform(0.4,0.6)
               z = random.uniform(0.4,0.6) # entre 0 y 1

               self.normal = torch.tensor([0.0,1.0,0.0])
               self.center = torch.tensor([x,y,z])
               abs_vector = np.linalg.norm(self.normal)
               if (abs_vector==0):  abs_vector = np.finfo(self.normal.numpy().dtype).eps
               self.normal = self.normal / abs_vector
          else:
              x = random.uniform(0.4,0.6)
              y = random.uniform(0.4,0.6)
              z = random.uniform(0.4,0.6) # entre 0 y 1

              n1 = random.random()
              n2 = random.random()
              n3 = random.random()

              # self.center = torch.tensor([0.5,y,0.5])
              # self.normal = torch.tensor([0.0,1.0,0.0])
              self.center = torch.tensor([x,y,z])
              self.normal = torch.tensor([n1,n2,n3])
              abs_vector = np.linalg.norm(self.normal)
              if (abs_vector==0):  abs_vector = np.finfo(self.normal.numpy().dtype).eps
              self.normal = self.normal / abs_vector

        else:

            if self.only_move:
                self.center = torch.tensor([0.5, 0.5,0.5])
                self.normal = torch.tensor([0.0,1.0,0.0])
            else:
                # during testing start sample a plane around the center point
                #self.center = torch.tensor([0.5, 0.5,0.5])
                self.center = torch.tensor([0.5, 0.5,0.5])
                #self.center = torch.tensor([0.5, 0.5,0.2])
                #self.normal = torch.tensor([0.00001, 0.00001,1.])
                #self.normal = torch.tensor([0.,1.,0.]) # Sagital
                if self.Plane == 'AAsc':
                    #self.normal = torch.tensor([1.,1.,0.]) # Axial
                    #self.normal = torch.tensor([0.,0.,1.]) # Axial
                    #self.normal = torch.tensor([1.,1.,-0.5]) # Axial
                    self.normal = torch.tensor([-0.5,1.,1]) # Axial
                    # self.center = torch.tensor([0.5, 0.5,0.3])
                    #self.normal = torch.tensor([-0.44876591,0.094530165,0.534079343])
                elif self.Plane == 'PAP' or self.Plane == 'PAD':
                    self.normal = torch.tensor([0.,1.,0.]) # Sagital
                    if new_try:
                        if self.tries<=1:
                            self.normal = torch.tensor([0.,0.,1.]) # Sagital
                        elif self.tries<=2:
                            self.normal = torch.tensor([1.,0.,0.]) # Sagital
                        elif self.tries<=3:
                            self.normal = torch.tensor([1.,1.,1.]) # Sagital
                elif self.Plane == 'LPA':
                    self.normal = torch.tensor([0.,1.,0.5]) # Sagital
                    if new_try:
                        if self.tries<=1:
                            self.normal = torch.tensor([0.,0.,1.]) # Sagital
                        elif self.tries<=2:
                            self.normal = torch.tensor([1.,0.,0.]) # Sagital
                        elif self.tries<=3:
                            self.normal = torch.tensor([1.,1.,1.]) # Sagital
                elif self.Plane == 'RPA':
                    self.normal = torch.tensor([1.,1.,0.]) # Coronal
                    if new_try:
                        if self.tries<=1:
                            self.normal = torch.tensor([0.,0.,1.]) # Sagital
                        elif self.tries<=2:
                            self.normal = torch.tensor([1.,0.,0.]) # Sagital
                        elif self.tries<=3:
                            self.normal = torch.tensor([1.,1.,1.]) # Sagital
                else:
                    self.normal = torch.tensor([1.,0.,0.]) # Coronal


        if self.only_move:
            self.XQN_lab,self.YQN_lab,self.ZQN_lab,self.VI_lab,self.D_lab,self.D_lab_real,_,self.V_lab_interp = stack_slice_selection(torch.unsqueeze(self.x,0),
                                                  torch.unsqueeze(self.y[3:6],0),centers= torch.unsqueeze(self.y[0:3]*2 -1,0),interp = False,slice_size =self.screen_dims[1],
                                                  Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                                  velocities=self.velocities,n_slices=self.screen_dims[0],
                                                  vel_interp=self.vel_interp,
                                                  PC_vel= self.PC_vel,
                                                  orto_2D = self.orto_2D,
                                                  ori_interp = self.ori_interp,
                                                  e_V = self.e_V)
            self.origin_lab = (self.y[0:3]*2 -1)*(self.Volume_realsize/2)
        else:
            self.XQN_lab,self.YQN_lab,self.ZQN_lab,self.VI_lab,self.D_lab,self.D_lab_real,self.origin_lab,self.V_lab_interp= stack_slice_selection(torch.unsqueeze(self.x,0),
                                                  torch.unsqueeze(self.y[3:6],0),centers= torch.unsqueeze(self.y[0:3]*2 -1,0),interp = False,slice_size =self.screen_dims[1],
                                                  Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                                  velocities=self.velocities,n_slices=self.screen_dims[0],
                                                  vel_interp=self.vel_interp,
                                                  PC_vel= self.PC_vel,
                                                  orto_2D = self.orto_2D,
                                                  ori_interp = self.ori_interp,
                                                  e_V = self.e_V)
            #self.origin_lab = (self.y[0:3]*2 -1)*(self.Volume_realsize/2)

        self.normal_angles = torch.acos(self.normal)

        if self.only_move:
            self.XQN,self.YQN,self.ZQN,self.VI,self.D,self.D_real,_,self.V_interp= stack_slice_selection(torch.unsqueeze(self.x,0),torch.unsqueeze(self.normal,0),
                                                  centers= torch.unsqueeze(self.center*2 - 1,0),interp = False, slice_size =self.screen_dims[1],
                                                  Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                                  velocities=self.velocities,n_slices=self.screen_dims[0],
                                                  vel_interp=self.vel_interp,
                                                  PC_vel= self.PC_vel,
                                                  orto_2D = self.orto_2D,
                                                  ori_interp = self.ori_interp,
                                                  e_V = self.e_V )
            self.origin = (self.center*2 -1)*(self.Volume_realsize/2)
        else:
            self.XQN,self.YQN,self.ZQN,self.VI,self.D,self.D_real,self.origin,self.V_interp = stack_slice_selection(torch.unsqueeze(self.x,0),torch.unsqueeze(self.normal,0),
                                                  centers= torch.unsqueeze(self.center*2 - 1,0),interp = False, slice_size =self.screen_dims[1],
                                                  Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                                  velocities=self.velocities,n_slices=self.screen_dims[0],
                                                  vel_interp=self.vel_interp,
                                                  PC_vel= self.PC_vel,
                                                  orto_2D = self.orto_2D,
                                                  ori_interp = self.ori_interp,
                                                  e_V = self.e_V )
            #self.origin = (self.center*2 -1)*(self.Volume_realsize/2)



        #print(self.D_lab_real)
        self._plane_lab_params = torch.cat((self.normal_lab_angles,torch.squeeze(self.D_lab_real).view(1)),dim=-1)
        self._plane_params = torch.cat((self.normal_angles,
                                        torch.squeeze(self.D_real).view(1)),dim=-1)

        self.cur_dist = calcDistParams(self._plane_lab_params,self._plane_params,0) + origins_distance(self.origin_lab,self.origin)*self.scale_d

        self.VI = map_to_uint8(self.VI,velocities=self.velocities,vel_interp=self.vel_interp)
        self.VI_lab = map_to_uint8(self.VI_lab,velocities=self.velocities,vel_interp=self.vel_interp)

        # print(self.VI.min(),self.VI.max())
        # print(self.VI_lab.min(),self.VI_lab.max())
        # print(self.x.min(),self.x.max())
        # exit()

        self.plane_state = {'XQN':self.XQN,'YQN':self.YQN,'ZQN':self.ZQN,
                            'VI':self.VI.numpy(),'plane_params':self._plane_params}


        # from matplotlib import pyplot as plt
        #
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(np.squeeze(self.VI[0,4,:,:].numpy()))
        # plt.subplot(1,2,2)
        # plt.imshow(np.squeeze(self.VI_lab[0,4,:,:].numpy()))
        # plt.show(block=True)

    def step(self, act, qvalues=0):
        """The environment's step function returns exactly what we need.
        Args:
          action:
        Returns:
          observation (object):
            an environment-specific object representing your observation of the environment. For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
          reward (float):
            amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.
          done (boolean):
            whether it's time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
          info (dict):
            diagnostic information useful for debugging. It can sometimes be useful for learning (for example, it might contain the raw probabilities behind the environment's last state change). However, official evaluations of your agent are not allowed to use this for learning.
        """

        self.terminal = False
        self._qvalues = qvalues

        # get current plane params
        current_plane_params = torch.clone(self._plane_params)
        next_plane_params = torch.clone(current_plane_params)
        # ---------------------------------------------------------------------


        # theta x+ (param a)
        if not self._supervised:
            if self.discrete:
                M = torch.eye(3,dtype=torch.float32)
                if (act==0):M = Rot3D_matrix(0,np.deg2rad(self.action_angle_step))
                # theta y+ (param b)
                if (act==1):M = Rot3D_matrix(1,np.deg2rad(self.action_angle_step))
                # theta z+ (param c)
                if (act==2):M = Rot3D_matrix(2,np.deg2rad(self.action_angle_step))
                # dist d+
                if (act==3):self.origin = self.origin + (self.action_dist_step * self.normal)/1000
                # theta x- (param a)
                if (act==4):M = Rot3D_matrix(0,-np.deg2rad(self.action_angle_step))
                # theta y- (param b)
                if (act==5):M = Rot3D_matrix(1,-np.deg2rad(self.action_angle_step))
                # theta z- (param c)
                if (act==6):M = Rot3D_matrix(2,-np.deg2rad(self.action_angle_step))
                # dist d-
                if (act==7):self.origin = self.origin - (self.action_dist_step * self.normal)/1000
            else:
                #M = torch.eye(3,dtype=torch.float32)
                if not self.only_move:
                    M1 = Rot3D_matrix(0,np.deg2rad(act[0]*self.action_angle_step))
                    # theta y+ (param b)
                    M2 = Rot3D_matrix(1,np.deg2rad(act[1]*self.action_angle_step))
                    # theta z+ (param c)
                    M3 = Rot3D_matrix(2,np.deg2rad(act[2]*self.action_angle_step))
                    # dist d+
                if self.only_move:
                    M1 = torch.eye(3)
                    M2 = torch.eye(3)
                    M3 = torch.eye(3)

                # self.origin[0] = self.origin[0] + (act[3]*self.action_dist_step)/1000
                # self.origin[1] = self.origin[1] + (act[4]*self.action_dist_step)/1000
                # self.origin[2] = self.origin[2] + (act[5]*self.action_dist_step)/1000

                self.origin_prev = torch.clone(self.origin)

                if self.only_move:
                    self.origin[0] = self.origin[0] + (act[0]*self.action_dist_step)/1000
                    self.origin[1] = self.origin[1] + (act[1]*self.action_dist_step)/1000
                    self.origin[2] = self.origin[2] + (act[2]*self.action_dist_step)/1000
                else:
                #     self.origin[0] = self.origin[0] + (act[3]*self.action_dist_step)/1000
                #     self.origin[1] = self.origin[1] + (act[4]*self.action_dist_step)/1000
                #     self.origin[2] = self.origin[2] + (act[5]*self.action_dist_step)/1000
                    self.origin = self.origin + (act[3]*self.action_dist_step * self.normal)/1000

                # self.origin[0


        if self._supervised and self.train and self.discrete:
            pass
        #     ## supervised
        #     dist_queue = deque(maxlen=self.actions)
        #     M_queue = deque(maxlen=self.actions)
        #     plane_params_queue = deque(maxlen=self.actions)
        #
        #     origin2 = self.origin
        #     # theta x+ (param a)
        #     next_plane_params = torch.clone(self._plane_params)
        #     M = Rot3D_matrix(0,np.deg2rad(self.action_angle_step))
        #     M_queue.append(M)
        #     normal2 = torch.matmul(M,self.normal)
        #     abs_vector = np.linalg.norm(normal2)
        #     if (abs_vector==0):  abs_vector = np.finfo(normal2.numpy().dtype).eps
        #     normal2 = normal2 / abs_vector
        #     next_plane_params[0:3] = torch.acos(normal2)
        #     plane_params_queue.append(next_plane_params)
        #     #print(0,self._plane_params,next_plane_params,self._plane_lab_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # theta y+ (param b) ----------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     M = Rot3D_matrix(1,np.deg2rad(self.action_angle_step))
        #     M_queue.append(M)
        #     normal2 = torch.matmul(M,self.normal)
        #     abs_vector = np.linalg.norm(normal2)
        #     if (abs_vector==0):  abs_vector = np.finfo(normal2.numpy().dtype).eps
        #     normal2 = normal2 / abs_vector
        #
        #     next_plane_params[0:3] = torch.acos(normal2)
        #     plane_params_queue.append(next_plane_params)
        #     #print(1,self._plane_params,next_plane_params,self._plane_lab_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # theta z+ (param c) ----------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     M = Rot3D_matrix(2,np.deg2rad(self.action_angle_step))
        #     M_queue.append(M)
        #     normal2 = torch.matmul(M,self.normal)
        #     abs_vector = np.linalg.norm(normal2)
        #     if (abs_vector==0):  abs_vector = np.finfo(normal2.numpy().dtype).eps
        #     normal2 = normal2 / abs_vector
        #     next_plane_params[0:3] = torch.acos(normal2)
        #     plane_params_queue.append(next_plane_params)
        #     #print(2,self._plane_params,next_plane_params,self._plane_lab_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # dist d+ ---------------------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     origin2 = self.origin + (self.action_dist_step * self.normal)/1000
        #
        #     _,_,_,_,_,self.D_real,origin2 = stack_slice_selection(torch.unsqueeze(self.x,0),
        #                                     torch.unsqueeze(self.normal,0),slice_size =self.screen_dims[1],
        #                                      centers = torch.unsqueeze((origin2*2) / self.Volume_realsize,0),interp = False,
        #                                       Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),
        #                                       voxels_size = torch.unsqueeze(self.spacing,0)*1e-3)
        #
        #     next_plane_params[3] = self.D_real
        #     M_queue.append(torch.eye(3))
        #     plane_params_queue.append(next_plane_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # theta x- (param a) ----------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     M = Rot3D_matrix(0,-np.deg2rad(self.action_angle_step))
        #     M_queue.append(M)
        #     normal2 = torch.matmul(M,self.normal)
        #     abs_vector = np.linalg.norm(normal2)
        #     if (abs_vector==0):  abs_vector = np.finfo(normal2.numpy().dtype).eps
        #     normal2 = normal2 / abs_vector
        #     next_plane_params[0:3] = torch.acos(normal2)
        #     plane_params_queue.append(next_plane_params)
        #     #print(4,self._plane_params,next_plane_params,self._plane_lab_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # theta y- (param b) ----------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     M = Rot3D_matrix(1,-np.deg2rad(self.action_angle_step))
        #     M_queue.append(M)
        #     normal2 = torch.matmul(M,self.normal)
        #     abs_vector = np.linalg.norm(normal2)
        #     if (abs_vector==0):  abs_vector = np.finfo(normal2.numpy().dtype).eps
        #     normal2 = normal2 / abs_vector
        #     next_plane_params[0:3] = torch.acos(normal2)
        #     plane_params_queue.append(next_plane_params)
        #     #print(5,self._plane_params,next_plane_params,self._plane_lab_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # theta z- (param c) ----------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     M = Rot3D_matrix(2,-np.deg2rad(self.action_angle_step))
        #     M_queue.append(M)
        #     normal2 = torch.matmul(M,self.normal)
        #     abs_vector = np.linalg.norm(normal2)
        #     if (abs_vector==0):  abs_vector = np.finfo(normal2.numpy().dtype).eps
        #     normal2 = normal2 / abs_vector
        #     next_plane_params[0:3] = torch.acos(normal2)
        #     plane_params_queue.append(next_plane_params)
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     #print(6,self._plane_params,next_plane_params,self._plane_lab_params)
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # dist d- ---------------------------------------------------------
        #     next_plane_params = torch.clone(self._plane_params)
        #     origin2 = self.origin - (self.action_dist_step * self.normal)/1000
        #
        #     _,_,_,_,_,self.D_real,origin2 = stack_slice_selection(torch.unsqueeze(self.x,0),
        #                                     torch.unsqueeze(self.normal,0),slice_size =self.screen_dims[1],
        #                                      centers = torch.unsqueeze((origin2*2) / self.Volume_realsize,0),interp = False,
        #                                       Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),
        #                                       voxels_size = torch.unsqueeze(self.spacing,0)*1e-3)
        #
        #     next_plane_params[3] = self.D_real
        #
        #     M_queue.append(torch.eye(3))
        #     plane_params_queue.append(next_plane_params)
        #
        #     # dist_queue.append(calcMeanDistTwoPlanes(self._groundTruth_plane.points, plane_params_queue[-1].points))
        #     dist_queue.append(calcDistParams(self._plane_lab_params, plane_params_queue[-1],self.scale_d))
        #
        #     # -----------------------------------------------------------------
        #     # get best plane based on lowest distance to the target
        #     next_plane_idx = np.argmin(dist_queue)
        #
        #     next_plane_params = plane_params_queue[next_plane_idx]
        #     M = M_queue[next_plane_idx]
        #
        #     self.reward = self._calc_reward_params(current_plane_params,
        #                                        next_plane_params)
        #     # threshold reward between -1 and 1
        #     self.reward = np.sign(self.reward)
        #
        #
        #     self.normal_angles = next_plane_params[0:3]
        #     #print(*dist_queue,'action',next_plane_idx)
        #
        #     if (next_plane_idx == 3):
        #         self.origin = self.origin + (self.action_dist_step * self.normal)/1000
        #     elif (next_plane_idx == 7):
        #         #print('h',self.origin)
        #         self.origin = self.origin - (self.action_dist_step * self.normal)/1000
        #         #print('h',self.origin)
        #
        #     #print(torch.acos(self.normal))
        #     #self.normal = torch.cos(self.normal_angles)
        #
        #     self.normal = torch.matmul(M,self.normal)
        #     #print(next_plane_idx,self.normal_angles,torch.acos(self.normal))
        #     dist_D, dist_angs = calcDistParams2(self._plane_lab_params,next_plane_params,self.scale_d)
        #     #print('DISTS ',next_plane_idx,dist_D,dist_angs)
        #
        #
        #     abs_vector = np.linalg.norm(self.normal)
        #     if (abs_vector==0):  abs_vector = np.finfo(self.normal.numpy().dtype).eps
        #     self.normal = self.normal / abs_vector
        #
        #     next_plane_params[0:3] = torch.acos(self.normal)
        #
        #     self.XQN,self.YQN,self.ZQN, VI,self.D,self.D_real,self.origin = stack_slice_selection(torch.unsqueeze(self.x,0),
        #                                     torch.unsqueeze(self.normal,0),slice_size =self.screen_dims[1],
        #                                      centers = torch.unsqueeze((self.origin*2) / self.Volume_realsize,0),interp = False,
        #                                       Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),
        #                                       voxels_size = torch.unsqueeze(self.spacing,0)*1e-3)
        #     next_plane_params[3] = self.D_real
        #
        #     self.VI = map_to_uint8(torch.squeeze(VI))

        else:

            if self.discrete:
                if (act == 0) or (act == 1) or (act == 2) or (act == 4) or (act == 5) or (act == 6):
                    normal = torch.matmul(M,self.normal)
                    if torch.cdist(normal.unsqueeze(0),self.normal.unsqueeze(0)).item() < 1e-4:
                        act-=1
                        if (act==0):M = Rot3D_matrix(0,np.deg2rad(self.action_angle_step))
                        # theta y+ (param b)
                        if (act==1):M = Rot3D_matrix(1,np.deg2rad(self.action_angle_step))
                        # theta z+ (param c)
                        if (act==-1):M = Rot3D_matrix(2,np.deg2rad(self.action_angle_step))
                        # theta x- (param a)
                        if (act==4):M = Rot3D_matrix(0,-np.deg2rad(self.action_angle_step))
                        # theta y- (param b)
                        if (act==5):M = Rot3D_matrix(1,-np.deg2rad(self.action_angle_step))
                        # theta z- (param c)
                        if (act==3):M = Rot3D_matrix(2,-np.deg2rad(self.action_angle_step))
                        self.normal = torch.matmul(M,self.normal)
                    else:
                        self.normal = normal
                    abs_vector = np.linalg.norm(self.normal)
                    if (abs_vector==0):  abs_vector = np.finfo(self.normal.numpy().dtype).eps
                    self.normal = self.normal / abs_vector
                    next_plane_params[0:3] = torch.acos(self.normal)
            else:
                normal = torch.matmul(M3,torch.matmul(M2,torch.matmul(M1,self.normal)))

                self.normal = normal
                abs_vector = np.linalg.norm(self.normal)
                if (abs_vector==0):  abs_vector = np.finfo(self.normal.numpy().dtype).eps
                self.normal = self.normal / abs_vector
                next_plane_params[0:3] = torch.acos(self.normal)

            if self.only_move:
                self.XQN,self.YQN,self.ZQN,VI,self.D,self.D_real,_,self.V_interp= stack_slice_selection(torch.unsqueeze(self.x,0),
                                        torch.unsqueeze(self.normal,0),slice_size =self.screen_dims[1],
                                    centers = torch.unsqueeze((self.origin*2)/ self.Volume_realsize,0),interp = False,
                                    Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),
                                    voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                    velocities=self.velocities,n_slices=self.screen_dims[0],
                                    vel_interp=self.vel_interp,
                                    PC_vel= self.PC_vel,
                                    orto_2D = self.orto_2D,
                                    ori_interp = self.ori_interp,
                                    e_V = self.e_V )
            else:
                if self.cnt < 149:
                    e_V_aux = None
                else:
                    e_V_aux = self.e_V
                self.XQN,self.YQN,self.ZQN,VI,self.D,self.D_real,self.origin,self.V_interp= stack_slice_selection(torch.unsqueeze(self.x,0),
                                        torch.unsqueeze(self.normal,0),slice_size =self.screen_dims[1],
                                    centers = torch.unsqueeze((self.origin*2)/ self.Volume_realsize,0),interp = False,
                                    Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),
                                    voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                    velocities=self.velocities,n_slices=self.screen_dims[0],
                                    vel_interp=self.vel_interp,
                                    PC_vel= self.PC_vel,
                                    orto_2D = self.orto_2D,
                                    ori_interp = self.ori_interp,
                                    e_V = e_V_aux)

            #next_plane_params[3] = torch.cdist(self.origin.unsqueeze(0),self.origin_lab.unsqueeze(0))*self.scale_d
            next_plane_params[3] = self.D_real
            if self.discrete:
                self.reward = self._calc_reward_params(current_plane_params,next_plane_params,VI_prev=self.VI.numpy(),VI_next=map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp).numpy(),MI=False)
            else:
                if self.exp_reward:
                    self.reward = self._calc_exp_reward_params(next_plane_params)
                else:
                    if self.orto_2D:
                        if self.only_move:
                            self.reward = self._calc_reward_params(current_plane_params,next_plane_params,VI_prev=self.VI.numpy(),VI_next=map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp).numpy(),MI=True,origin_prev=self.origin_prev,origin_next=self.origin)
                        else:
                            #self.reward = self._calc_reward_params(current_plane_params,next_plane_params,VI_prev=self.VI.numpy(),VI_next=map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp).numpy(),MI=True,origin_prev=self.origin_prev,origin_next=self.origin)
                            self.reward = self._calc_reward_params(current_plane_params,next_plane_params,VI_prev=self.VI.numpy()[0,:,:],VI_next=map_to_uint8(VI[0,:,:],velocities=self.velocities,vel_interp=self.vel_interp).numpy(),MI=True)#,origin_prev=self.origin_prev,origin_next=self.origin)
                    else:
                        self.reward = self._calc_reward_params(current_plane_params,next_plane_params,VI_prev=self.VI.numpy(),VI_next=map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp).numpy(),MI=True)
                    #self.reward = self._ard_params(current_plane_params,next_plane_params)

            # threshold reward between -1 and 1
            if self.discrete:
                self.reward = np.sign(self.reward)
            else:
                self.reward = np.clip(self.reward,-2.0,2.0)

            go_out = False


            ## MODIFY!!!
            # -----------------------------------------------------------------
            # check if the screen is not full of zeros (background)
            if self.velocities and not self.vel_interp:
                go_out = checkBackgroundRatio(map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp)[0,self.screen_dims[0]//2,:,:], min_pixel_val=0.001, ratio=0.95)
            else:
                if not self.vel_interp:

                    go_out = checkBackgroundRatio(map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp)[self.screen_dims[0]//2,:,:], min_pixel_val=0.001, ratio=0.95)
                if self.PC_vel:
                    go_out = checkBackgroundRatio(map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp)[0,self.screen_dims[0]//2,:,:], min_pixel_val=0.001, ratio=0.95)

            # # also check if go out (sampling from outside the volume)
            # # by checking if the new origin
            # if not go_out:
            #     go_out = checkOriginLocation(self.sitk_image,next_plane.origin)
            # also check if plane parameters got very high
            # if not go_out:
            #     go_out = checkParamsBound(next_plane.params,
            #                               self._groundTruth_plane.params)
            # punish lowest reward if the agent tries to go out and keep same plane
            if go_out:
                self.reward = -3 # lowest possible reward
                #next_plane = copy.deepcopy(self._plane)
                # print("ME SALI")
                # exit()
                if self.train:
                    self.terminal = True # end episode and restart
                # else:
                #     self.cnt = 0
                #
            else:
              self.VI = map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp)



        # ---------------------------------------------------------------------
        # update current plane
        self._plane_params = copy.deepcopy(next_plane_params)
        self._total_plane_history.append(self._plane_params)
        # terminate if maximum number of steps is reached
        self.cnt += 1
        if self.cnt >= self.max_num_frames: self.terminal = True
        # check oscillation and reduce action step or terminate if minimum
        oscillate = False
        if self._oscillate:
            oscillate = True
            self.oscillations+=1
            if self.discrete:
                if self.train and self._supervised:
                    self._plane_params,self.normal,self.origin = self.getBestPlaneTrain()
                else:
                    self._plane_params,self.normal,self.origin = self.getBestPlane()
                self._update_heirarchical()

            # find distance metrics

            self.cur_dist = calcDistParams(self._plane_lab_params,self._plane_params,self.scale_d)

            self._clear_history()
            # terminate if distance steps are less than 1
            #if self.action_dist_step < 0.1:   self.terminal = True

        # ---------------------------------------------------------------------
        # find distance error
        # _, dist = zip(*[projectPointOnPlane(point, self._plane.norm, self._plane.origin) for point in self.landmarks_gt])
        # self.cur_dist = np.mean(np.abs(dist))
        self.cur_dist_params =  calcDistParams(self._plane_lab_params,self._plane_params,0) + origins_distance(self.origin_lab,self.origin)*self.scale_d
        dist_D, dist_angs = calcDistParams2(self._plane_lab_params,self._plane_params,1e3)
        self.cur_dist = self.cur_dist_params

        origins_error = origins_distance(self.origin,self.origin_lab)

        self.current_episode_score.feed(self.reward)
        self._update_history() # store results in memory
        if self._supervised:
          self._dist_supervised_history.append(np.min(dist_queue))

        # dot_norms = np.clip(self.normal_lab.numpy().dot(self.normal.numpy()),-1.0,1.0)
        # angle_between_norms = np.rad2deg(np.arccos(dot_norms))

        dot_norms = np.clip(np.abs(self.normal_lab.numpy().dot(self.normal.numpy())),-1.0,1.0)
        #dot_norms = np.abs(self.normal_lab.numpy().dot(self.normal.numpy()))
        #print(dot_norms,self.normal_lab,self.normal)
        if not self.only_move:
            angle_between_norms = np.rad2deg(np.arccos(dot_norms))
        else:
            angle_between_norms = 0


        # terminate if distance between params are low during training
        if self.discrete:
            if self.done_action:
                if act == 8 :
                    self.terminal = True
                if act == 8  and  (origins_error*1e3<3) and (angle_between_norms<=5):
                    self.terminal = True
                    self.reward = 5
                    self.num_success.feed(1)
            else:
                if self.train  and (origins_error*1e3<3) and (angle_between_norms<=5):
                    self.terminal = True
                    self.reward = self.reward + 3
                    self.num_success.feed(1)
        else:
            if not self.ori_finish:
                if self.train and (origins_error*1e3<3):
                    if self.reward > 0:
                        self.n_finish +=1
                        if self.n_finish >= 3 :
                            self.ori_finish = True
                        self.reward = self.reward + 1
            else:
                if self.train and (origins_error*1e3<3) and (angle_between_norms<=5):
                    if self.reward > 0:
                        self.n_finish +=1
                        if self.n_finish >= 10 :
                            self.terminal = True
                            self.num_success.feed(1)
                        self.reward = self.reward + 1


        info = {'score': self.current_episode_score.sum, 'gameOver': self.terminal,
                'distError': self.cur_dist, 'distAngle': angle_between_norms,
                'Patient':self.Patient,'angle_step':self.action_angle_step,'dist_step':self.action_dist_step,
                'plane_params': self._plane_params,'plane_lab_params': self._plane_lab_params,
                'dist_D':dist_D,'dist_angs':dist_angs,
                'origins_error':origins_error,'origin':self.origin,
                'D_real':self.D_real,'Oscillations':self.oscillations,'Oscillate':oscillate,
                'go_out':go_out}

        if self.NUMPY:
          self.plane_state = self.VI.numpy()
          #self.plane_state = np.expand_dims(np.squeeze(self.VI.numpy()[4,:,:]),0)
        else:
          self.plane_state = {'XQN':self.XQN,'YQN':self.YQN,'ZQN':self.ZQN,
                            'VI':self.VI.numpy()}
        self._current_state = (self.plane_state, self.reward, self.terminal, info)
        return self.plane_state, self.reward, self.terminal, info


    def return_BestPlane(self):

        best_idx = np.argmin(self._bestq_history)
        # best_idx = np.argmax(self._bestq_history)
        #self._plane_history[best_idx]
        normal = self._normal_history[best_idx]
        origin = self._origin_history[best_idx]

        _,_,_,VI,_,_,_ = stack_slice_selection(torch.unsqueeze(self.x,0),
                                torch.unsqueeze(normal,0),slice_size =self.screen_dims[1],
                            centers = torch.unsqueeze((origin*2)/ self.Volume_realsize,0),interp = False,
                            Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),
                            voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                            velocities=self.velocities,n_slices=self.screen_dims[0],
                            vel_interp=self.vel_interp,
                            PC_vel= self.PC_vel,
                            orto_2D = self.orto_2D )

        return np.squeeze(map_to_uint8(VI,velocities=self.velocities,vel_interp=self.vel_interp).numpy()),best_idx

    def get_data_lists(self):
        return self.train_list,self.valid_list,self.test_list

    def get_voxel_size(self):
        return self.spacing

    def get_origin_lab(self):
        return self.origin_lab
    def get_normal_lab(self):
        return self.normal_lab

    def get_normal(self):
        return self.normal

    def get_lab_plane(self):
        plane_lab_state = {'XQN':self.XQN_lab,'YQN':self.YQN_lab,'ZQN':self.ZQN_lab,
                            'VI':self.VI_lab.numpy()}
        return plane_lab_state

    def get_Patient(self):
        return self.Patient

    def get_volume(self):
      return self.x, self.Volume_realsize,self.SEG

    def get_line_on_plane(self,origin=None,normal=None,switch=False,limits=[0,1],limits_lab=[-1,1]):

        XQN,YQN,ZQN,VI,_,_,_,XQN_line,YQN_line,ZQN_line,VI_line,XQN_lab,YQN_lab,ZQN_lab,VI_lab = line_selection(torch.unsqueeze(self.x,0),torch.unsqueeze(self.y[3:6],0),
                                              centers= torch.unsqueeze(self.y[0:3]*2 -1,0),interp = False, slice_size =self.screen_dims[1],
                                              Volumes_realsize = torch.unsqueeze(self.Volume_realsize,0),voxels_size = torch.unsqueeze(self.spacing,0)*1e-3,
                                              velocities=self.velocities,n_slices=self.screen_dims[0],
                                              normals2 = torch.unsqueeze(normal,0),
                                              centers2 = torch.unsqueeze((origin*2)/ self.Volume_realsize,0),
                                              switch=switch,
                                              limits=limits,
                                              limits_lab=limits_lab,
                                              orto_2D=self.orto_2D,
                                              PC_vel=self.PC_vel)
        # if self.velocities:
        #     VI = VI[0,:,:,:]
        #     VI_line = VI_line[0,:,:,:]
        #     VI_lab = VI_lab[0,:,:,:]

        return  XQN,YQN,ZQN,VI,XQN_line,YQN_line,ZQN_line,VI_line,XQN_lab,YQN_lab,ZQN_lab,VI_lab


    def _update_history(self):
        ''' update history buffer with current state
        '''
        # update location history
        self._loc_history[:-1] = self._loc_history[1:]
        #loc = self._plane.origin
        loc = self._plane_params
        # logger.info('loc {}'.format(loc))
        self._loc_history[-1] = (np.around(loc[0].numpy(),decimals=2),
                                 np.around(loc[1].numpy(),decimals=2),
                                 np.around(loc[2].numpy(),decimals=2),
                                 np.around(loc[3].numpy(),decimals=3))
        # update distance history
        self._dist_history.append(self.cur_dist)
        self._dist_history_params.append(self.cur_dist_params)
        # update params history
        self._plane_history.append(self._plane_params)
        self._origin_history.append(self.origin)
        self._normal_history.append(self.normal)
        self._bestq_history.append(np.max(self._qvalues))
        # update q-value history
        self._qvalues_history[:-1] = self._qvalues_history[1:]
        self._qvalues_history[-1] = self._qvalues

    def _update_heirarchical(self):
        self.action_angle_step = np.maximum(self.action_angle_step - 1,1)
        self.action_dist_step = np.maximum(self.action_dist_step - 1,1)
        #self.action_dist_step = np.maximum(self.action_dist_step - 0.001,0)


    def _clear_history(self):
        self._plane_history.clear()
        self._normal_history.clear()
        self._origin_history.clear()
        self._bestq_history.clear()
        self._dist_history.clear()
        self._dist_history_params.clear()
        self._dist_supervised_history.clear()
        # self._loc_history = [(0,) * self.dims] * self._history_length
        self._loc_history = [(0,) * 4] * self._history_length
        self._qvalues_history = [(0,) * self.actions] * self._history_length


    @property
    def _oscillate(self):
        ''' Return True if the agent is stuck and oscillating
        '''
        counter = Counter(self._loc_history)
        freq = counter.most_common()
        # return false is history is empty (begining of the game)
        if len(freq) < 2: return False
        # check frequency
        if freq[0][0] == (0,0,0,0):
            if (freq[1][1]>2):
                #print(freq)
                # logger.info('oscillating {}'.format(self._loc_history))
                return True
            else:
                return False
        elif (freq[0][1]>2):
            #print(freq)
            # logger.info('oscillating {}'.format(self._loc_history))
            return True

    def getBestPlane(self):
        ''' get best location with best qvalue from last for locations
        stored in history
        '''
        best_idx = np.argmin(self._bestq_history)
        # best_idx = np.argmax(self._bestq_history)
        return self._plane_history[best_idx],self._normal_history[best_idx],self._origin_history[best_idx]

    def getBestPlaneTrain(self):
        ''' get best location with best qvalue from last for locations
        stored in history
        '''
        best_idx = np.argmin(self._dist_supervised_history)
        #print(best_idx)
        # best_idx = np.argmax(self._bestq_history)
        return self._plane_history[best_idx],self._normal_history[best_idx],self._origin_history[best_idx]

    def _calc_reward_params(self, prev_params, next_params,VI_prev=None,VI_next=None,MI=False,origin_prev=None,origin_next=None ):
        ''' Calculate the new reward based on the euclidean distance to the target plane
        '''
        # if VI_prev is not None:
        #     prev_ssim = ssim(np.squeeze(VI_prev[4,:,:]),np.squeeze(self.VI_lab.numpy()[4,:,:]))
        # else:
        #     prev_ssim = 1.0

        if MI:
            if self.velocities and not self.vel_interp:
                hist_2d, x_edges, y_edges = np.histogram2d(VI_prev[0,:,:,:].ravel(),self.VI_lab.numpy()[0,:,:,:].ravel(),bins=64)
                MI_IP = mutual_information(hist_2d)
                MI_V = 0
                #for ch in [1,2,3]:
                for ch in [1,2]:
                    hist_2d, x_edges, y_edges = np.histogram2d(VI_prev[ch,:,:,:].ravel(),self.VI_lab.numpy()[ch,:,:,:].ravel(),bins=64)
                    MI_V += mutual_information(hist_2d)

                #prev_MI = np.minimum((MI_IP + MI_V/3)/2,1.)
                prev_MI = MI_IP + MI_V/2

            else:
                if self.orto_2D:
                    MI_P = 0
                    #for ch in [0,1,2]:
                    for ch in [0]:
                        hist_2d, x_edges, y_edges = np.histogram2d(VI_prev.ravel(),self.VI_lab.numpy()[ch,:,:].ravel(),bins=64)
                        MI_P += mutual_information(hist_2d)
                    prev_MI = np.minimum(MI_P/3,1.)
                    #prev_MI = np.minimum(MI_P,1.)
                else:
                    hist_2d, x_edges, y_edges = np.histogram2d(VI_prev.ravel(),self.VI_lab.numpy().ravel(),bins=64)
                    prev_MI = np.minimum(mutual_information(hist_2d),1.)


        else:
            prev_MI = 1.0

        if origin_prev is not None:
            if self.only_move:
                prev_dist = origins_distance(self.origin_lab,origin_prev)*self.scale_d
            else:
                #prev_dist = calcDistParams(self._plane_lab_params, prev_params,scale_d=0) + origins_distance(self.origin_lab,origin_prev)*self.scale_d
                if not self.ori_finish:
                    prev_dist =  origins_distance(self.origin_lab,origin_prev)*self.scale_d
                else:
                    prev_dist = calcDistParams(self._plane_lab_params, prev_params,scale_d=0) + 2*(1.0 - prev_MI) + origins_distance(self.origin_lab,origin_prev)*(self.scale_d/2)
        else:
            prev_dist = calcDistParams(self._plane_lab_params, prev_params,self.scale_d) + 2*(1.0 - prev_MI)


        # if VI_next is not None:
        #     next_ssim = ssim(np.squeeze(VI_next[4,:,:]),np.squeeze(self.VI_lab.numpy()[4,:,:]))
        # else:
        #     next_ssim = 1.0

        if MI:
            if self.velocities and not self.vel_interp:
                hist_2d, x_edges, y_edges = np.histogram2d(VI_next[0,:,:,:].ravel(),self.VI_lab.numpy()[0,:,:,:].ravel(),bins=64)
                MI_IP = mutual_information(hist_2d)
                MI_V = 0
                #for ch in [1,2,3]:
                for ch in [1,2]:
                    hist_2d, x_edges, y_edges = np.histogram2d(VI_next[ch,:,:,:].ravel(),self.VI_lab.numpy()[ch,:,:,:].ravel(),bins=64)
                    MI_V += mutual_information(hist_2d)
                #next_MI = np.minimum((MI_IP + MI_V/3)/2,1.)
                next_MI = MI_IP + MI_V/2
            else:
                if self.orto_2D:
                    MI_P = 0
                    #for ch in [0,1,2]:
                    for ch in [0]:
                        hist_2d, x_edges, y_edges = np.histogram2d(VI_next.ravel(),self.VI_lab.numpy()[ch,:,:].ravel(),bins=64)
                        MI_P += mutual_information(hist_2d)
                    next_MI = np.minimum(MI_P/3,1.)
                    #next_MI = np.minimum(MI_P,1.)
                else:
                    hist_2d, x_edges, y_edges = np.histogram2d(VI_next.ravel(),self.VI_lab.numpy().ravel(),bins=64)
                    next_MI = np.minimum(mutual_information(hist_2d),1.0)
        else:
            next_MI = 1.0

        if origin_next is not None:
            if self.only_move:
                next_dist = origins_distance(self.origin_lab,origin_next)*self.scale_d
            else:
                if not self.ori_finish:
                    next_dist = origins_distance(self.origin_lab,origin_next)*self.scale_d
                else:
                    next_dist = calcDistParams(self._plane_lab_params, next_params,0) + 2*(1. - next_MI) + origins_distance(self.origin_lab,origin_next)*(self.scale_d/2)
                #next_dist = calcDistParams(self._plane_lab_params, next_params,0) + origins_distance(self.origin_lab,origin_next)*self.scale_d
        else:
            next_dist = calcDistParams(self._plane_lab_params, next_params,self.scale_d) + 2*(1. - next_MI)

        return prev_dist - next_dist

    def _calc_exp_reward_params(self,next_params,alpha = 10):
        ''' Calculate the new reward based on the euclidean distance to the target plane
        '''
        dist = calcDistParams(self._plane_lab_params, next_params,self.scale_d)

        exp_dist = np.exp(-np.power(dist,2)/alpha)

        return exp_dist

    def reset_stat(self):
        """ Reset all statistics counter"""
        self.stats = defaultdict(list)
        self.num_games = StatCounter()
        self.num_success = StatCounter()
