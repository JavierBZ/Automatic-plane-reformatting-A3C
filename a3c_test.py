from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from Medical_env_def import MedicalPlayer
from a3c_utils import setup_logger, process_state
from a3c_models import A3C_CONV, A3C_MLP, A3C_CONV_new, A3C_CONV_2
from a3c_player_util import Agent
import time
import logging
import pickle
from data_loaders import get_loaders
import os
import wandb

from collections import deque

def test(args, shared_model,group,job_type,shared_model2=None):
    ptitle('Test Agent')

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    if args.gdrive:
        wb_method = "thread"
    else:
        wb_method = "fork"
    #with wandb.init(project='Automatic reformatting', entity='javierbz',config = vars(args),settings=wandb.Settings(start_method=wb_method)):
    with wandb.init(project='Automatic reformatting', entity='javierbz',config = vars(args),group=group,job_type=job_type,settings=wandb.Settings(start_method=wb_method)):
        args = wandb.config
        gpu_id = args.gpu_ids[-1]
        log = {}
        setup_logger('{}_log'.format(args.env),
                     r'{0}{1}_log'.format(args.log_dir, args.env))
        log['{}_log'.format(args.env)] = logging.getLogger(
            '{}_log'.format(args.env))
        d_args = vars(args)
        for k in d_args.keys():
            log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

        torch.manual_seed(args.seed)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed)
        env =  MedicalPlayer(data_loader = 'valid',
                        max_num_frames = args.max_episode_length,
                        train = False,
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
                        PC_vel = args.PC_vel,
                        folder = args.folder,
                        orto_2D = args.orto_2D,
                        only_move = args.only_move
                         )

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
            dummy_model = A3C_MLP(num_channels, num_actions)
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
                dummy_model = A3C_CONV(num_channels, num_actions,out_feats=args.out_feats,layers=args.layers)
            else:
                player.model = A3C_CONV_new(num_channels, num_actions)
                dummy_model = A3C_CONV_new(num_channels, num_actions)

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
                dummy_model = A3C_CONV_2(num_channels, num_actions)
            else:
                player.model = A3C_CONV_new(num_channels, num_actions)
                dummy_model = A3C_CONV_new(num_channels, num_actions)


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
                player.model = player.model.cuda()
                player.state = player.state.cuda()
        player.model.eval()
        max_score = 0
        min_angle = 10_000
        min_dist = 10_000
        volume = 0

        reward_sum = 0
        start_time = time.time()
        num_tests = 0

        _,valid_list,_ = player.env.get_data_lists()
        N_volumes = len(valid_list)


        reward_totals = np.zeros((N_volumes,))
        TotalDistErrors = np.zeros((N_volumes,))
        AngleErrors = np.zeros((N_volumes,))
        D_errors = np.zeros((N_volumes,))
        OriginsErrors  = np.zeros((N_volumes,))

        reward_total_L = []
        TotalDistError_L = []
        AngleError_L = []
        OriginsError_L  = []

        dummy_input = None

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
            if (volume == (N_volumes)):

                reward_totals = np.zeros((N_volumes,))
                TotalDistErrors = np.zeros((N_volumes,))
                AngleErrors = np.zeros((N_volumes,))
                D_errors = np.zeros((N_volumes,))
                OriginsErrors  = np.zeros((N_volumes,))
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
                volume = 0
                num_tests+=1
                time.sleep(90)

            if player.done:

                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.model.load_state_dict(shared_model.state_dict())
                else:
                    player.model.load_state_dict(shared_model.state_dict())


            player.action_test()
            reward_sum += player.reward

            if player.done:
                reward_totals[volume] = reward_sum
                TotalDistError   = player.info['distError']
                AngleError = player.info['distAngle']
                D_error = player.info['dist_angs']
                OriginsError = player.info['origins_error']*args.scale_d

                TotalDistErrors[volume] = TotalDistError
                AngleErrors[volume] = AngleError
                D_errors[volume] = D_error
                OriginsErrors[volume] = OriginsError
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

                volume+=1

            if player.done and volume == (N_volumes):

                reward_mean = np.mean(reward_totals)

                TotalDistError_mean   = np.mean(TotalDistErrors)
                AngleError_mean = np.mean(AngleErrors)
                D_error_mean = np.mean(D_errors)
                OriginsError_mean = np.mean(OriginsErrors)

                log['{}_log'.format(args.env)].info(
                    "Time {0}, episode length {1}, reward mean {2:.4f}, Total_distM {3:.3f}, Angle ErrorM {4:.2f}, OriginsErrorM {5:.3f},D_errorM {6:.3f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        player.eps_len, reward_mean,TotalDistError_mean,AngleError_mean,OriginsError_mean,D_error_mean))

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                wandb.log({'reward mean':reward_mean,
                            'Total Dist mean': TotalDistError_mean,
                            'Angle error mean': AngleError_mean,
                            'Origins error mean':OriginsError_mean,
                            'D error mean':D_error_mean,
                            'score':wandb.Histogram(reward_totals),
                            'Total Dist': wandb.Histogram(TotalDistErrors),
                            'Angle error': wandb.Histogram(AngleErrors),
                            'Origins error': wandb.Histogram(OriginsErrors),
                            'D error': wandb.Histogram(D_errors)
                            },
                            step = epoch_mins)



                # log['{}_log'.format(args.env)].info(
                #     "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, Total_dist {4:.3f}, Angle Error {5:.2f}, OriginsError {6:.3f}, num_tests {7}  ".
                #     format(
                #         time.strftime("%Hh %Mm %Ss",
                #                       time.gmtime(time.time() - start_time)),
                #         reward_sum, player.eps_len, reward_mean,TotalDistError,AngleError,OriginsError,num_tests))

                reward_total_L.append(reward_mean)
                TotalDistError_L.append(TotalDistError_mean)
                AngleError_L.append(AngleError_mean)
                OriginsError_L.append(OriginsError_mean)
                if num_tests >= 0 and (num_tests%10 == 0):
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):

                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{0}{1}_{2}.dat'.format(args.save_model_dir, args.env,str(time.ctime()).replace(':', '_')))

                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_{2}.dat'.format(args.save_model_dir, args.env,str(time.ctime()).replace(':', '_')))


                    info = {'reward_total_L':reward_total_L,
                            'TotalDistError_L':TotalDistError_L,
                            'AngleError_L':AngleError_L,
                            'OriginsError_L':OriginsError_L}

                    if args.gdrive:
                        path_info = args.log_dir + "%s_%sinfo.pickle" %("Medical_env", str(time.ctime()).replace(':', '_'))
                    else:
                        path_info = "/home1/jebisbal"
                        path_info = path_info + "/Results_a3c/" + "%s_%sinfo.pickle" %("Medical_env", str(time.ctime()).replace(':', '_'))

                    # Saving the objects:
                    with open(path_info, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(info, f)

                if args.save_max and reward_mean >= max_score:
                    max_score = reward_mean
                    if args.velocities and not args.vel_interp:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance',
                                    'Param D error','Angle error','Origin error',
                                    'Label Vx','Net Vx','Label Vy','Net Vy','Label Vz','Net Vz']
                    elif args.PC_vel:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance',
                                    'Param D error','Angle error','Origin error',
                                    'Label V','Net V']
                    elif args.orto_2D:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance',
                                    'Param D error','Angle error','Origin error',
                                    'Label orto1','Net orto1','Label orto2','Net orto2']

                    else:
                        columns = ['Patient','Label Image','Net Image','Score','Params distance','Param D error','Angle error','Origin error']

                    Log_table = wandb.Table(columns=columns)
                    for p in range(len(patients)):
                        if args.velocities and not args.vel_interp:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],OriginsErrors[p],
                                    wandb.Image(vx_labs[p]),wandb.Image(last_vx[p]),
                                    wandb.Image(vy_labs[p]),wandb.Image(last_vy[p]),
                                    wandb.Image(vz_labs[p]),wandb.Image(last_vz[p]))
                        elif args.PC_vel:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],OriginsErrors[p],
                                    wandb.Image(v_labs[p]),wandb.Image(last_v[p]))
                        elif args.orto_2D:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],OriginsErrors[p],
                                    wandb.Image(orto1_labs[p]),wandb.Image(last_orto1[p]),
                                    wandb.Image(orto2_labs[p]),wandb.Image(last_orto2[p]))

                        else:
                            Log_table.add_data(patients[p],wandb.Image(plane_labs[p]),wandb.Image(last_images[p]),reward_totals[p],
                                    TotalDistErrors[p],D_errors[p],AngleErrors[p],OriginsErrors[p])

                    # wandb.log({'Images':[wandb.Image(last_image, caption = patient) for last_image, patient in zip(last_images,patients)]
                    #             },
                    #             step = num_tests)
                    wandb.log({'Tabla':Log_table},step = epoch_mins)
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))
                            torch.save(state_to_save,os.path.join(wandb.run.dir, "best_model.pt"))
                        dummy_model.load_state_dict(player.model.state_dict())
                        if dummy_input is None:
                            if args.lstm:
                                if args.frame_history > 1 or args.TwoDim or (args.velocities and not args.vel_interp) or args.PC_vel:
                                    dummy_input = (torch.clone(player.state.squeeze().unsqueeze(0).detach().cpu()),(torch.clone(player.hx.detach().cpu()), torch.clone(player.cx.detach().cpu())))
                                else:
                                    dummy_input = (torch.clone(player.state.unsqueeze(0).unsqueeze(0).detach().cpu()),(torch.clone(player.hx.detach().cpu()), torch.clone(player.cx.detach().cpu())))

                            else:
                                if args.frame_history > 1 or args.TwoDim or (args.velocities and not args.vel_interp) or args.PC_vel:
                                    dummy_input = torch.clone(player.state.squeeze().unsqueeze(0).detach().cpu())
                                else:
                                    dummy_input = torch.clone(player.state.unsqueeze(0).unsqueeze(0).detach().cpu())
                        if args.lstm:
                            torch.onnx.export(dummy_model,
                                            (dummy_input,),
                                            #'{0}{1}_{2}.onnx'.format(args.save_model_dir, args.env,str(time.ctime()).replace(':', '_')),
                                            os.path.join(wandb.run.dir, "best_model.onnx"),
                                            input_names=['input','h0&c0'],
                                            output_names=['value', 'mu', 'sigma','hn&cn'])
                                            #dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
                        else:
                            torch.onnx.export(dummy_model,
                                            dummy_input,
                                            #'{0}{1}_{2}.onnx'.format(args.save_model_dir, args.env,str(time.ctime()).replace(':', '_')),
                                            os.path.join(wandb.run.dir, "best_model.onnx"),
                                            input_names=['input'],
                                            output_names=['value', 'mu', 'sigma'])
                                            #dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
                        wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))
                        wandb.save(os.path.join(wandb.run.dir, "best_model.onnx"))
                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))

                        dummy_model.load_state_dict(player.model.state_dict())
                        if dummy_input is None:
                            if args.velocities and not args.vel_interp  or args.PC_vel:
                                dummy_input = (torch.clone(player.state.squeeze().unsqueeze(0).detach().cpu()),(torch.clone(player.hx.detach().cpu()), torch.clone(player.cx.detach().cpu())))
                            else:
                                dummy_input = (torch.clone(player.state.unsqueeze(0).unsqueeze(0).detach().cpu()),(torch.clone(player.hx.detach().cpu()), torch.clone(player.cx.detach().cpu())))
                            #dummy_input = (torch.clone(player.state.unsqueeze(0).detach().cpu()),(torch.clone(player.hx.detach().cpu()), torch.clone(player.cx.detach().cpu())))
                        torch.onnx.export(dummy_model,
                                        (dummy_input,),
                                        #'{0}{1}_{2}.onnx'.format(args.save_model_dir, args.env,str(time.ctime()).replace(':', '_')),
                                        os.path.join(wandb.run.dir, "a3c_model.onnx"),
                                        input_names=['input','h0&c0'],
                                        output_names=['value', 'mu', 'sigma','hn&cn'])
                                        #dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})

                        wandb.save(os.path.join(wandb.run.dir, "best_model.onnx"))
                if  AngleError_mean <= min_angle:
                    min_angle = AngleError_mean
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{0}{1}_MinAngle.dat'.format(args.save_model_dir, args.env))
                            torch.save(state_to_save,os.path.join(wandb.run.dir, '{0}_MinAngle.dat'.format(args.env)))
                        wandb.save(os.path.join(wandb.run.dir, '{0}_MinAngle.dat'.format(args.env)))
                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_MinAngle.dat'.format(args.save_model_dir, args.env))
                if  OriginsError_mean <= min_dist:
                    min_dist = OriginsError_mean
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{0}{1}_MinDist.dat'.format(args.save_model_dir, args.env))
                            torch.save(state_to_save,os.path.join(wandb.run.dir, '{0}_MinDist.dat'.format(args.env)))
                        wandb.save(os.path.join(wandb.run.dir, '{0}_MinDist.dat'.format(args.env)))
                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_MinDist.dat'.format(args.save_model_dir, args.env))

            if player.done:
                reward_sum = 0
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
