'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
modified from (batch, h, w, ch) to (batch, ch, h, w)
'''

import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        # print(self._encode_observation(10).shape)
        # print(self._encode_observation(0).shape)
        # print(self._encode_observation(0)[None].shape)
        # print(self._encode_observation(10)[None].shape)
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8, where observation[i*img_c:(i+1)*img_c, :, :]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            #return np.concatenate(frames, 0) # c, h, w instead of h, w c
            return np.concatenate(np.expand_dims(frames,0), 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            # c, h, w instead of h, w c
            img_s , img_h, img_w =self.obs.shape[1], self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1,img_s, img_h, img_w)
            #return self.obs[start_idx:end_idx].reshape(-1,img_h, img_w)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # if observation is an image...
        # if len(frame.shape) > 1:
        #     # transpose image frame into c, h, w instead of h, w, c
        #     frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame
        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done



if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import torch
    from data_loaders import get_loaders

    from Medical_env import MedicalPlayer

    train_list1 = ["000a","001a","002a","003a","004a","005a","006","007a","008a","010a","011a","012b","013a"]
    train_list2 = ["043","044","045","047","048","049","051","053","055"]
    train_list3 = ["083","085","086","087"]
    train_list4 = ["150","151","154","155","160",]


    valid_list1 = ["014a","017a"]
    valid_list2 = ["046","052"]
    valid_list3 = []
    valid_list4 = ["152"]

    test_list1 = ["016a","015a"]
    test_list2 = ["054","050"]
    test_list3 = ["091"]
    test_list4 = ["157"]

    train_list = train_list1 + train_list2 + train_list3 + train_list4
    valid_list = valid_list1 + valid_list2 + valid_list3 + valid_list4
    test_list = test_list1 + test_list2 + test_list3 + test_list4

    test_list = ["016a","054"]

    data_path = "/Volumes/HV620S/Data_Javier2/Data_Drive/"
    labels_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plane_Labels/"

    train_loader, valid_loader, test_loader = get_loaders(train_list,valid_list,test_list,data_path=data_path,labels_path=labels_path)

    # CUDA variables
    USE_CUDA = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    env =  MedicalPlayer(data_loader = test_loader,max_num_frames = 300,
                  train = True,screen_dims = (9,70,70),spacing = None,
                  history_length=30,supervised = False, scale_d = 10,
                  angle_step = 4, dist_step =0.01,NUMPY=True )

    # create replay buffer
    replay_buffer_size = 1_000
    frame_history_len = 4
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # plane_lab = env.get_lab_plane()
    # Vol,Volume_realsize = env.get_volume()

    total_steps = 999

    rewards = []
    terminals = []
    infos = []
    last_obs = env._reset_loader()
    for step in tqdm(range(total_steps),position=0,leave=True):
        # store last frame, returned idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()
        #print(observations.shape)
        action = random.randint(0,7)
        qvalue = 0

        obs, reward, done, info = env.step(action,qvalue)

        rewards.append(reward)
        terminals.append(done)
        infos.append(info)
        # store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        if done:
          obs = env._reset()

        # update last_obs
        last_obs = obs

    print(replay_buffer.obs[0].shape)
    batch_size = 4
    if replay_buffer.can_sample(batch_size):
        # sample transition batch from replay memory
        # done_mask = 1 if next state is end of episode
        obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
        obs_t = torch.from_numpy(obs_t).type(dtype) / 255.0
        act_t = torch.from_numpy(act_t).type(dlongtype)
        rew_t = torch.from_numpy(rew_t).type(dtype)
        obs_tp1 = torch.from_numpy(obs_tp1).type(dtype) / 255.0
        done_mask = torch.from_numpy(done_mask).type(dtype)

    print(obs_t.shape)

    plt.figure(figsize=(20,10))
    i = 1
    for sample in range(batch_size):
        for h in range(frame_history_len):
          plt.subplot(batch_size,4,i)
          #plt.imshow(np.squeeze(obs_t.cpu().numpy()[sample,4 + h*9,:,:]),extent=[0,obs_t.shape[2],0,obs_t.shape[2]],aspect="auto")
          plt.imshow(np.squeeze(obs_t.cpu().numpy()[sample,h,4,:,:]),extent=[0,obs_t.shape[2],0,obs_t.shape[2]],aspect="auto")
          #plt.imshow(np.squeeze(obs_t.cpu().numpy()[sample,h,:,:]),extent=[0,obs_t.shape[2],0,obs_t.shape[2]],aspect="auto")
          plt.title('history ' + str(h))
          i+=1
          plt.colorbar()

    plt.tight_layout()
    plt.show(block=True)
