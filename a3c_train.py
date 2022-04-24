from setproctitle import setproctitle as ptitle
import numpy as np
import torch
import torch.optim as optim
from Medical_env_def import MedicalPlayer
from a3c_utils import ensure_shared_grads, process_state
from a3c_models import A3C_CONV, A3C_MLP, A3C_CONV_new,A3C_CONV_2
from a3c_player_util import Agent

from data_loaders import get_loaders
import wandb
import time
from collections import deque


def train(rank, args, shared_model, optimizer,group,job_type,logging,shared_model2=None):
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    if logging:
        if args.gdrive:
            wb_method = "thread"
        else:
            wb_method = "fork"
        with wandb.init(project='Automatic reformatting', entity='javierbz',config = vars(args),group=group,job_type=job_type,settings=wandb.Settings(start_method=wb_method)):
            ptitle('Training Agent: {}'.format(rank))


            gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
            torch.manual_seed(args.seed + rank)
            if gpu_id >= 0:
                torch.cuda.manual_seed(args.seed + rank)

            env =  MedicalPlayer(data_loader = 'train',
                            max_num_frames = args.max_episode_length,
                            train = True,
                            screen_dims = args.screen_dims,
                            spacing = None,
                            history_length=30,
                            supervised = False,
                            scale_d = args.scale_d,
                            angle_step = args.angle_step,
                            dist_step =args.dist_step,
                            NUMPY=True,
                            discrete=False,
                            Drive=args.gdrive,
                            cluster=args.cluster,
                            contrast=args.contrast,
                            Plane = args.Plane,
                            velocities = args.velocities,
                            vel_interp = args.vel_interp,
                            PC_vel = args.PC_vel,
                            folder = args.folder,
                            orto_2D = args.orto_2D,
                            only_move = args.only_move
                            )

            env.seed(args.seed + rank)
            player = Agent(None, env, args, None)
            player.gpu_id = gpu_id
            if args.model == 'MLP':
                num_actions = args.n_actions
                if (args.velocities and not args.vel_interp) or args.TwoDim:
                    num_channels= args.frame_history * 3
                elif args.PC_vel:
                    num_channels =args.frame_history * 2
                else:
                    num_channels= args.frame_history
                player.model = A3C_MLP(num_channels, num_actions)
            if args.model == 'CONV':
                num_actions = args.n_actions
                if (args.velocities and not args.vel_interp) or args.TwoDim:
                    num_channels= args.frame_history * 3
                elif args.PC_vel:
                    num_channels =args.frame_history * 2
                else:
                    num_channels= args.frame_history
                if args.lstm:
                    player.model = A3C_CONV(num_channels, num_actions,out_feats=args.out_feats,layers=args.layers)
                else:
                    player.model = A3C_CONV_new(num_channels, num_actions)
            if args.model == 'CONV_2':
                num_actions = args.n_actions
                if (args.velocities and not args.vel_interp) or args.TwoDim:
                    num_channels= args.frame_history * 3
                elif args.PC_vel:
                    num_channels =args.frame_history * 2
                else:
                    num_channels= args.frame_history
                if args.lstm:
                    player.model = A3C_CONV_2(num_channels, num_actions)
                else:
                    player.model = A3C_CONV_new(num_channels, num_actions)

            wandb.watch(player.model,log='all',log_freq=100)


            if args.frame_history > 1:
                player.states = deque([],maxlen=args.frame_history)
                for fill in range(args.frame_history-1):
                    state = np.zeros((9,70,70),dtype=np.int8)
                    player.states.append(state)
                state = player.env._reset_loader()
                player.states.append(state)
                player.state = process_state(torch.as_tensor(player.states).float(),velocities=args.velocities,vel_interp=args.vel_interp)
            else:
                state = player.env._reset_loader()
                player.state = process_state(torch.from_numpy(state).float(),velocities=args.velocities,vel_interp=args.vel_interp)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
                    player.model = player.model.cuda()
            player.model.train()
            train_list,_,_ = player.env.get_data_lists()
            t_log = 0
            t_log_freq =  len(train_list)
            reward_sum = 0
            reward_totals = np.zeros((t_log_freq,))
            TotalDistErrors = np.zeros((t_log_freq,))
            AngleErrors = np.zeros((t_log_freq,))
            D_errors = np.zeros((t_log_freq,))

            last_images = []
            if args.velocities and not args.vel_interp:
                last_vx = []
                last_vy = []
                last_vz = []
            if args.PC_vel:
                last_v = []
            if args.orto_2D:
                last_orto1 = []
                last_orto2 = []
            patients = []
            plane_labs = []
            if args.velocities and not args.vel_interp:
                vx_labs = []
                vy_labs = []
                vz_labs = []
            if args.PC_vel:
                v_labs = []
            if args.orto_2D:
                orto1_labs = []
                orto2_labs = []

            start_time = time.time()
            while True:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.model.load_state_dict(shared_model.state_dict())
                else:
                    player.model.load_state_dict(shared_model.state_dict())
                if player.done:
                    if args.lstm:
                        if gpu_id >= 0:
                            with torch.cuda.device(gpu_id):
                                player.cx = torch.zeros(1, args.out_feats).cuda()
                                player.hx = torch.zeros(1, args.out_feats).cuda()
                        else:
                            player.cx = torch.zeros(1, args.out_feats)
                            player.hx = torch.zeros(1, args.out_feats)
                else:
                    if args.lstm:
                        player.cx = player.cx.data
                        player.hx = player.hx.data

                # if not player.done_1:
                #     while not player.done_1 or player.done:
                #         for step in range(args.num_steps*2):
                #             player.action_test()
                #         if not player.done:
                #             player.clear_actions()
                # if player.done_1:
                #     player.clear_actions()
                for step in range(args.num_steps):
                    player.action_train()
                    reward_sum += player.rewards[-1]
                    if player.done:
                        reward_totals[t_log] = reward_sum
                        TotalDistError   = player.info['distError']
                        AngleError = player.info['distAngle']
                        D_error = player.info['dist_angs']

                        TotalDistErrors[t_log] = TotalDistError
                        AngleErrors[t_log] = AngleError
                        D_errors[t_log] = D_error

                        if args.frame_history > 1:
                            last_images.append(player.state.detach().cpu()[-1,args.mid_slice,:,:])
                        else:
                            if args.velocities and not args.vel_interp:
                                last_images.append(player.state.detach().cpu()[0,args.mid_slice,:,:])
                                last_vx.append(player.state.detach().cpu()[1,args.mid_slice,:,:])
                                last_vy.append(player.state.detach().cpu()[2,args.mid_slice,:,:])
                                last_vz.append(player.state.detach().cpu()[2,args.mid_slice,:,:])
                            elif args.PC_vel:
                                l_state = player.state.detach().cpu()[0,args.mid_slice,:,:].numpy()
                                l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                                last_images.append(l_state)
                                l_state = player.state.detach().cpu()[1,args.mid_slice,:,:].numpy()
                                l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                                last_v.append(l_state)
                            elif args.orto_2D:
                                l_state = player.state.detach().cpu()[0,:,:].numpy()
                                l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                                last_images.append(l_state)
                                l_state = player.state.detach().cpu()[1,:,:].numpy()
                                l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                                last_orto1.append(l_state)
                                l_state = player.state.detach().cpu()[2,:,:].numpy()
                                l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                                last_orto2.append(l_state)
                            else:
                                l_state = player.state.detach().cpu()[args.mid_slice,:,:].numpy()
                                l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                                last_images.append(l_state)
                        patients.append(player.env.get_Patient())

                        plane_lab = player.env.get_lab_plane()
                        if args.velocities and not args.vel_interp:
                            plane_labs.append(plane_lab['VI'][0,args.mid_slice,:,:])
                            vx_labs.append(plane_lab['VI'][1,args.mid_slice,:,:])
                            vy_labs.append(plane_lab['VI'][2,args.mid_slice,:,:])
                            vz_labs.append(plane_lab['VI'][2,args.mid_slice,:,:])
                        elif args.PC_vel:
                            l_state = plane_lab['VI'][0,args.mid_slice,:,:]
                            l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                            plane_labs.append(l_state)
                            l_state = plane_lab['VI'][1,args.mid_slice,:,:]
                            l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                            v_labs.append(l_state)
                        elif args.orto_2D:
                            l_state = plane_lab['VI'][0,:,:]
                            l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                            plane_labs.append(l_state)
                            l_state = plane_lab['VI'][1,:,:]
                            l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                            orto1_labs.append(l_state)
                            l_state = plane_lab['VI'][2,:,:]
                            l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                            orto2_labs.append(l_state)
                        else:
                            l_state = plane_lab['VI'][args.mid_slice,:,:]
                            l_state =  (((l_state - l_state.min())/(l_state.max() - l_state.min()))*255).astype(np.uint8)
                            plane_labs.append(l_state)

                        reward_sum = 0
                        t_log+=1
                        break

                if player.done:
                    player.eps_len = 0
                    if args.frame_history > 1:
                        player.states = deque([],maxlen=args.frame_history)
                        for fill in range(args.frame_history-1):
                            state = np.zeros((9,70,70),dtype=np.int8)
                            player.states.append(state)
                        state = player.env.reset()
                        player.states.append(state)
                        player.state = process_state(torch.as_tensor(player.states).float(),velocities=args.velocities,vel_interp=args.vel_interp)
                    else:
                        state = player.env.reset()
                        player.state = process_state(torch.from_numpy(state).float(),velocities=args.velocities,vel_interp=args.vel_interp)
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            player.state = player.state.cuda()

                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        R = torch.zeros(1, 1).cuda()
                else:
                    R = torch.zeros(1, 1)
                if not player.done:
                    state = player.state
                    if args.model == 'CONV' or args.model == 'CONV_2':
                        if args.frame_history > 1 or args.TwoDim or (args.velocities and not args.vel_interp) or args.PC_vel:
                            state = state.squeeze().unsqueeze(0)
                        else:
                            state = state.unsqueeze(0).unsqueeze(0)
                    if args.lstm:
                        value, _, _, _ = player.model(
                            (state, (player.hx, player.cx)))

                    else:
                        value, _, _ = player.model(state)
                    R = value.data


                player.values.append(R)
                policy_loss = 0
                value_loss = 0
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        gae = torch.zeros(1, 1).cuda()
                else:
                    gae = torch.zeros(1, 1)
                for i in reversed(range(len(player.rewards))):
                    R = args.gamma * R + player.rewards[i]
                    advantage = R - player.values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    # Generalized Advantage Estimataion
            #          print(player.rewards[i])
                    delta_t = player.rewards[i] + args.gamma * \
                        player.values[i + 1].data - player.values[i].data

                    gae = gae * args.gamma * args.tau + delta_t

                    policy_loss = policy_loss - \
                        (player.log_probs[i].sum() * gae) - \
                        (0.01 * player.entropies[i].sum())



                player.model.zero_grad()
                (policy_loss + 0.7 * value_loss).backward()
                ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
                optimizer.step()
                player.clear_actions()



                if t_log >= (t_log_freq):

                    reward_mean = np.mean(reward_totals)
                    TotalDistError_mean   = np.mean(TotalDistErrors)
                    AngleError_mean = np.mean(AngleErrors)
                    D_error_mean = np.mean(D_errors)

                    end_time = time.time()
                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                    wandb.log({"Policy loss": policy_loss,
                                "Value loss": value_loss,
                                'Total Dist mean': TotalDistError_mean,
                                'Angle error mean': AngleError_mean,
                                'D error mean':D_error_mean,
                                'Mean score':reward_mean,
                                'Scores': reward_totals,
                                'Total Dist': wandb.Histogram(TotalDistErrors),
                                'Angle error': wandb.Histogram(AngleErrors),
                                'D error': wandb.Histogram(D_errors),
                                },step = epoch_mins)

                    if args.velocities and not args.vel_interp:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance',
                                    'Param D error','Angle error',
                                    'Label Vx','Net Vx','Label Vy','Net Vy','Label Vz','Net Vz']
                    elif args.PC_vel:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance',
                                    'Param D error','Angle error',
                                    'Label V','Net V']
                    elif args.orto_2D:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance',
                                    'Param D error','Angle error',
                                    'Label orto1','Net orto1','Label orto2','Net orto2']
                    else:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance','Param D error','Angle error']

                    Log_table = wandb.Table(columns=columns)
                    for p in range(len(patients)):
                        if args.velocities and not args.vel_interp:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],
                                    wandb.Image(vx_labs[p]),wandb.Image(last_vx[p]),
                                    wandb.Image(vy_labs[p]),wandb.Image(last_vy[p]),
                                    wandb.Image(vz_labs[p]),wandb.Image(last_vz[p]))
                        elif args.PC_vel:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],
                                    wandb.Image(v_labs[p]),wandb.Image(last_v[p]))
                        elif args.orto_2D:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],
                                    wandb.Image(orto1_labs[p]),wandb.Image(last_orto1[p]),
                                    wandb.Image(orto2_labs[p]),wandb.Image(last_orto2[p]))
                        else:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p])

                    wandb.log({'Tabla':Log_table})


                    reward_totals = np.zeros((t_log_freq,))
                    TotalDistErrors = np.zeros((t_log_freq,))
                    AngleErrors = np.zeros((t_log_freq,))
                    D_errors = np.zeros((t_log_freq,))

                    last_images = []
                    if args.velocities and not args.vel_interp:
                        last_vx = []
                        last_vy = []
                        last_vz = []
                    if args.PC_vel:
                        last_v = []
                    if args.orto_2D:
                        last_orto1 = []
                        last_orto2 = []
                    patients = []
                    if args.velocities and not args.vel_interp:
                        vx_labs = []
                        vy_labs = []
                        vz_labs = []
                    if args.PC_vel:
                        v_labs = []
                    if args.orto_2D:
                        orto1_labs = []
                        orto2_labs = []

                    t_log = 0


    else:


        ptitle('Training Agent: {}'.format(rank))


        gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
        torch.manual_seed(args.seed + rank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + rank)

        env =  MedicalPlayer(data_loader = 'train',
                        max_num_frames = args.max_episode_length,
                        train = True,
                        screen_dims = args.screen_dims,
                        spacing = None,
                        history_length=30,
                        supervised = False,
                        scale_d = args.scale_d,
                        angle_step = args.angle_step,
                        dist_step =args.dist_step,
                        NUMPY=True,
                        discrete=False,
                        Drive=args.gdrive,
                        cluster=args.cluster,
                        contrast=args.contrast,
                        Plane = args.Plane,
                        velocities=args.velocities,
                        vel_interp=args.vel_interp,
                        PC_vel=args.PC_vel,
                        folder = args.folder,
                        orto_2D = args.orto_2D,
                        only_move = args.only_move)

        env.seed(args.seed + rank)
        player = Agent(None, env, args, None)
        player.gpu_id = gpu_id
        if args.model == 'MLP':
            num_actions = args.n_actions
            if (args.velocities and not args.vel_interp) or args.TwoDim:
                num_channels= args.frame_history * 3
            elif args.PC_vel:
                num_channels =args.frame_history * 2
            else:
                num_channels= args.frame_history
            player.model = A3C_MLP(num_channels, num_actions)
        if args.model == 'CONV':
            num_actions = args.n_actions
            if (args.velocities and not args.vel_interp) or args.TwoDim:
                num_channels= args.frame_history * 3
            elif args.PC_vel:
                num_channels =args.frame_history * 2
            else:
                num_channels= args.frame_history
            if args.lstm:
                player.model = A3C_CONV(num_channels, num_actions,out_feats=args.out_feats,layers=args.layers)
            else:
                player.model = A3C_CONV_new(num_channels, num_actions)
        if args.model == 'CONV_2':
            num_actions = args.n_actions
            if args.velocities and not args.vel_interp:
                num_channels= args.frame_history * 3
            elif args.PC_vel:
                num_channels =args.frame_history * 2
            else:
                num_channels= args.frame_history
            if args.lstm:
                player.model = A3C_CONV_2(num_channels, num_actions)
            else:
                player.model = A3C_CONV_new(num_channels, num_actions)


        if args.frame_history > 1:
            player.states = deque([],maxlen=args.frame_history)
            for fill in range(args.frame_history-1):
                state = np.zeros((9,70,70),dtype=np.int8)
                player.states.append(state)
            player.state = player.env._reset_loader()
            player.states.append(state)
            player.state = process_state(torch.as_tensor(player.states).float(),velocities=args.velocities,vel_interp=args.vel_interp)
        else:
            state = player.env._reset_loader()
            player.state = process_state(torch.from_numpy(state).float(),velocities=args.velocities,vel_interp=args.vel_interp)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = player.state.cuda()
                player.model = player.model.cuda()
        player.model.train()


        while True:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            if player.done:
                if args.lstm:
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            player.cx = torch.zeros(1, args.out_feats).cuda()
                            player.hx = torch.zeros(1, args.out_feats).cuda()
                    else:
                        player.cx = torch.zeros(1, args.out_feats)
                        player.hx = torch.zeros(1, args.out_feats)
            else:
                if args.lstm:
                    player.cx = player.cx.data
                    player.hx = player.hx.data

            # if not player.done_1:
            #     while not player.done_1 or player.done:
            #         for step in range(args.num_steps*2):
            #             player.action_test()
            #         if not player.done:
            #             player.clear_actions()
            # if player.done_1:
            #     player.clear_actions()
            for step in range(args.num_steps):
                player.action_train()



            if player.done:
                player.eps_len = 0
                if args.frame_history > 1:
                    player.states = deque([],maxlen=args.frame_history)
                    for fill in range(args.frame_history-1):
                        state = np.zeros((9,70,70),dtype=np.int8)
                        player.states.append(state)
                    state = player.env.reset()
                    player.states.append(state)
                    player.state = process_state(torch.as_tensor(player.states).float(),velocities=args.velocities,vel_interp=args.vel_interp)
                else:
                    state = player.env.reset()
                    player.state = process_state(torch.from_numpy(state).float(),velocities=args.velocities,vel_interp=args.vel_interp)
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    R = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
            if not player.done:
                state = player.state
                if args.model == 'CONV' or args.model == 'CONV_2':
                    if args.frame_history > 1 or args.TwoDim or (args.velocities and not args.vel_interp) or args.PC_vel:
                        state = state.squeeze().unsqueeze(0)
                    else:
                        state = state.unsqueeze(0).unsqueeze(0)
                if args.lstm:
                    value, _, _, _ = player.model(
                        (state, (player.hx, player.cx)))

                else:
                    value, _, _ = player.model(state)
                R = value.data


            player.values.append(R)
            policy_loss = 0
            value_loss = 0
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    gae = torch.zeros(1, 1).cuda()
            else:
                gae = torch.zeros(1, 1)
            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
        #          print(player.rewards[i])
                delta_t = player.rewards[i] + args.gamma * \
                    player.values[i + 1].data - player.values[i].data

                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    (player.log_probs[i].sum() * gae) - \
                    (0.01 * player.entropies[i].sum())



            player.model.zero_grad()
            (policy_loss + 0.7 * value_loss).backward()
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()
            player.clear_actions()
