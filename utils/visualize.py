import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
import torch
import seaborn as sb
import pandas as pd



def visualize_3Dslice(volume,Xqn,Yqn,Zqn,Vi,SEG = None,path_fig = 'prueba.html',
                      origin1 = None,origin2=None,origin3=None,interp_size = [60,120,120],Volume_realsize = None,
                      stack = False,save =True,x2 = False, x3=False,
                      Xqn2=None,Yqn2=None,Zqn2=None,Vi2=None,
                      Xqn3=None,Yqn3=None,Zqn3=None,Vi3=None,
                      show_vol = True,
                      Xqn_lab=None,
                      Yqn_lab=None,
                      Zqn_lab=None,
                      Vi_lab=None,
                      no_volume=False
                      ):

      #volume = torch.squeeze(x)
      #values = volume

      if Volume_realsize is not None:
        x1 = np.linspace(-(Volume_realsize[1]/2), (Volume_realsize[1]/2), volume.shape[1])
        y1 = np.linspace(-(Volume_realsize[0]/2), (Volume_realsize[0]/2), volume.shape[0])
        z1 = np.linspace(-(Volume_realsize[2]/2), (Volume_realsize[2]/2), volume.shape[2])
      else:
        x1 = np.linspace(-1, 1, volume.shape[1])
        y1 = np.linspace(-1, 1, volume.shape[0])
        z1 = np.linspace(-1, 1, volume.shape[2])

      X, Y, Z = np.meshgrid(x1, y1, z1)



      if SEG is not None:

        #fig = go.Figure(data=go.Isosurface(
        fig = go.Figure(data=go.Volume(
          x=Y.flatten(),
          y=X.flatten(),
          z=Z.flatten(),
          value=SEG.flatten(),
          isomin=0.5,
          isomax=2,
          opacity=0.3,
          showscale = False,
          # surface_count=2, # number of isosurfaces, 2 by default: only min and max
          # colorbar_nticks=3, # colorbar ticks correspond to isosurface values
          caps=dict(x_show=False, y_show=False),
          #colorscale='viridis',
          #reversescale= True
          #opacity = 0.6,
          # uirevision = 'opacity'
          ))
        fig.update_layout(scene=dict(
                      xaxis=dict(visible=False),
                      yaxis=dict(visible=False),
                      zaxis=dict(visible=False),
                      )
                      )
      else:

        fig = go.Figure(data=go.Volume(
          x=Y.flatten(),
          y=X.flatten(),
          z=Z.flatten(),
          isomin=-2,
          isomax=2,
          value=volume.flatten(),
          opacity=0.3,
          showscale = False,
          #opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
          #colorscale='RdBu',
          caps= dict(x_show=False, y_show=False, z_show=False), # no caps
          ))

      # if center is not None:
      #   if Volume_realsize is not None:
      #     center = (center*2 - 1) * Volume_realsize/2
      #   else:
      #     center = center*2 - 1
      #   fig.add_trace(go.Scatter3d(x = np.array(center[0]), y = np.array(center[1]), z = np.array(center[2]),mode = 'markers',marker=dict(
      #         size=6,
      #         color='red',                # set color to an array/list of desired values
      #         #colorscale='Viridis',   # choose a colorscale
      #         opacity=0.8
      #     )))



      if no_volume:
          fig = go.Figure()
          fig.update_layout(scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        )
                        )

      if stack:
        for s in range(Xqn.shape[0]):
          fig.add_trace(go.Surface(z=Zqn[s,:,:], x=Xqn[s,:,:], y=Yqn[s,:,:], surfacecolor = Vi[s,:,:],opacity=0.8,colorscale='gray'  ))
      else:
        fig.add_trace(go.Surface(z=Zqn, x=Xqn, y=Yqn, surfacecolor = Vi,opacity=0.8,colorscale='gray',showscale=False ))

      if x2:
        if stack:
          for s in range(Xqn.shape[0]):
            fig.add_trace(go.Surface(z=Zqn2[s,:,:], x=Xqn2[s,:,:], y=Yqn2[s,:,:], surfacecolor = Vi2[s,:,:],opacity=0.8,colorscale='jet'  ))
        else:
          #fig.add_trace(go.Surface(z=Zqn2, x=Xqn2, y=Yqn2, surfacecolor = Vi2,opacity=0.8,colorscale='jet',showscale=False ))

          fig.add_trace(go.Scatter3d(x = Xqn2.flatten(), y = Yqn2.flatten(), z = Zqn2.flatten(),mode = 'markers',  marker=dict(
                  size=3,
                  color='red',#Vi2.flatten(),                # set color to an array/list of desired values
                 # colorscale='Viridis',   # choose a colorscale
                  opacity=0.8
              )))
      if x3:
        if stack:
          for s in range(Xqn.shape[0]):
            fig.add_trace(go.Surface(z=Zqn3[s,:,:], x=Xqn3[s,:,:], y=Yqn3[s,:,:], surfacecolor = Vi3[s,:,:],opacity=0.8,colorscale='jet'  ))
        else:
          #fig.add_trace(go.Surface(z=Zqn2, x=Xqn2, y=Yqn2, surfacecolor = Vi2,opacity=0.8,colorscale='jet',showscale=False ))

          fig.add_trace(go.Scatter3d(x = Xqn3.flatten(), y = Yqn3.flatten(), z = Zqn3.flatten(),mode = 'markers',  marker=dict(
                  size=3,
                  color='yellow',#Vi2.flatten(),                # set color to an array/list of desired values
                 # colorscale='Viridis',   # choose a colorscale
                  opacity=0.8
              )))
      if Xqn_lab is not None:
          if stack:
            for s in range(Xqn.shape[0]):
              fig.add_trace(go.Surface(z=Zqn_lab[s,:,:], x=Xqn_lab[s,:,:], y=Yqn_lab[s,:,:], surfacecolor = Vi_lab[s,:,:],opacity=0.8,colorscale='jet'  ))
          else:
            #fig.add_trace(go.Surface(z=Zqn_lab, x=Xqn_lab, y=Yqn_lab, surfacecolor = Vi_lab,opacity=0.8,colorscale='jet',showscale=False ))

            fig.add_trace(go.Scatter3d(x = Xqn_lab.flatten(), y = Yqn_lab.flatten(), z = Zqn_lab.flatten(),mode = 'markers',  marker=dict(
                    size=3,
                    color='blue',#Vi2.flatten(),                # set color to an array/list of desired values
                   # colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )))
      if origin1 is not None:
              fig.add_trace(go.Scatter3d(x = np.array(origin1[0]), y = np.array(origin1[1]), z = np.array(origin1[2]),mode = 'markers',marker=dict(
                    size=5,
                    color='cyan'               # set color to an array/list of desired values
                    #colorscale='Viridis',   # choose a colorscale
                    #opacity=0.95
                )))

      if origin2 is not None:
              fig.add_trace(go.Scatter3d(x = np.array(origin2[0]), y = np.array(origin2[1]), z = np.array(origin2[2]),mode = 'markers',marker=dict(
                    size=5,
                    color='cyan'              # set color to an array/list of desired values
                    #colorscale='Viridis',   # choose a colorscale
                   # opacity=0.95
                )))
      if origin3 is not None:
              fig.add_trace(go.Scatter3d(x = np.array(origin3[0]), y = np.array(origin3[1]), z = np.array(origin3[2]),mode = 'markers',marker=dict(
                    size=5,
                    color='cyan'              # set color to an array/list of desired values
                    #colorscale='Viridis',   # choose a colorscale
                   # opacity=0.95
                )))

      fig.update_traces(showlegend=True)
      fig.update_scenes(zaxis_autorange="reversed")
      #fig.update_scenes(zaxis_showgrid= False,xaxis_showgrid= False,yaxis_showgrid= False)
      #if Volume_realsize is not None:
        #fig.update_layout(scene_aspectratio=dict(x=Volume_realsize[0]*5, y=Volume_realsize[1]*5, z=Volume_realsize[2]*5))
      #fig.update_layout(scene_aspectratio=dict(x=1, y=2, z=2))
      if save:
        fig.write_html(path_fig)
      #fig.show()

      return fig


def visualize_3Dslice_full(Results,SEG = None,path_folder = 'prueba.html',
                    interp_size = [60,120,120],Volume_realsize = None,
                      stack = False,save =True,x2 = False,
                      origins=False,
                      n_patients = 0,
                      show_vol = True,
                      Planes = None,
                      mid_slice = None):

    #volume = torch.squeeze(x)
    #values = volume

    for data_idx in range(n_patients):

        Result = Results[0]

        volume = Result['Volumes'][data_idx]
        SEG = Result['SEGS'][data_idx]
        Volume_realsize = Result['Volumes_realsize'][data_idx]


        if Volume_realsize is not None:
            x1 = np.linspace(-(Volume_realsize[1]/2), (Volume_realsize[1]/2), volume.shape[1])
            y1 = np.linspace(-(Volume_realsize[0]/2), (Volume_realsize[0]/2), volume.shape[0])
            z1 = np.linspace(-(Volume_realsize[2]/2), (Volume_realsize[2]/2), volume.shape[2])
        else:
            x1 = np.linspace(-1, 1, volume.shape[1])
            y1 = np.linspace(-1, 1, volume.shape[0])
            z1 = np.linspace(-1, 1, volume.shape[2])

        X, Y, Z = np.meshgrid(x1, y1, z1)


        if SEG is not None:

            #fig = go.Figure(data=go.Isosurface(
            fig = go.Figure(data=go.Volume(
              x=Y.flatten(),
              y=X.flatten(),
              z=Z.flatten(),
              value=SEG.flatten(),
              isomin=0.5,
              isomax=2,
              opacity=0.6,
              showscale = False,
              # surface_count=2, # number of isosurfaces, 2 by default: only min and max
              # colorbar_nticks=3, # colorbar ticks correspond to isosurface values
              caps=dict(x_show=False, y_show=False),
              #colorscale='gray',
              #reversescale= True
              #opacity = 0.6,
              # uirevision = 'opacity'
              name = 'Segmentation'
              ))
            # fig.update_layout(scene=dict(
            #                  xaxis=dict(visible=False),
            #                  yaxis=dict(visible=False),
            #                  zaxis=dict(visible=False),
            #                  )
            #                  )
        else:
            # print("MAAAAAX",np.max(volume.flatten()))
            # print("MEANNNN",np.mean(volume.flatten()))
            # print("MIIIIINNNN",np.min(volume.flatten()))
            fig = go.Figure(data=go.Volume(
              x=Y.flatten(),
              y=X.flatten(),
              z=Z.flatten(),
              isomin=-2,
              isomax=2,
              value=volume.flatten(),
              opacity=0.8,
              showscale = False,
              #opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
              colorscale='viridis',
              reversescale=True,
              caps= dict(x_show=False, y_show=False, z_show=False), # no caps
              name = 'Volume'))
            fig.update_layout(scene=dict(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(visible=False),
                            )
                            )

        for plane,name in enumerate(Planes):
            Result = Results[plane]

            if stack:
                Xqn = np.squeeze(Result['plane_labs'][data_idx]['XQN'])
                Yqn = np.squeeze(Result['plane_labs'][data_idx]['YQN'])
                Zqn = np.squeeze(Result['plane_labs'][data_idx]['ZQN'])
                Vi = np.squeeze(Result['plane_labs'][data_idx]['VI'])

                Xqn2 = np.squeeze(Result['XQN_T'][data_idx,int(Result['Min_steps'][data_idx]),:,:,:])
                Yqn2 = np.squeeze(Result['YQN_T'][data_idx,int(Result['Min_steps'][data_idx]),:,:,:])
                Zqn2 = np.squeeze(Result['ZQN_T'][data_idx,int(Result['Min_steps'][data_idx]),:,:,:])
                Vi2 = np.squeeze(Result['VI_T'][data_idx,int(Result['Min_steps'][data_idx]),:,:,:])

            else:

                Xqn = np.squeeze(Result['plane_labs'][data_idx]['XQN'][mid_slice ,:,:])
                Yqn = np.squeeze(Result['plane_labs'][data_idx]['YQN'][mid_slice ,:,:])
                Zqn = np.squeeze(Result['plane_labs'][data_idx]['ZQN'][mid_slice ,:,:])
                Vi = np.squeeze(Result['plane_labs'][data_idx]['VI'][mid_slice ,:,:])

                Xqn2 = np.squeeze(Result['XQN_T'][data_idx,int(Result['Min_steps'][data_idx]),mid_slice ,:,:])
                Yqn2 = np.squeeze(Result['YQN_T'][data_idx,int(Result['Min_steps'][data_idx]),mid_slice ,:,:])
                Zqn2 = np.squeeze(Result['ZQN_T'][data_idx,int(Result['Min_steps'][data_idx]),mid_slice ,:,:])
                Vi2 = np.squeeze(Result['VI_T'][data_idx,int(Result['Min_steps'][data_idx]),mid_slice ,:,:])




            if origins:
                origin1 = Result['Origins_lab'][data_idx,:]
                origin2 = Result['Origins'][data_idx,int(Result['Min_steps'][data_idx]),:]

                if origin1 is not None:
                    fig.add_trace(go.Scatter3d(x = np.array(origin1[0]), y = np.array(origin1[1]), z = np.array(origin1[2]),mode = 'markers',marker=dict(
                          size=10,
                          color='red'               # set color to an array/list of desired values
                          #colorscale='Viridis',   # choose a colorscale
                          #opacity=0.95
                      ),name=name))

                if origin2 is not None:
                    fig.add_trace(go.Scatter3d(x = np.array(origin2[0]), y = np.array(origin2[1]), z = np.array(origin2[2]),mode = 'markers',marker=dict(
                          size=10,
                          color='blue'              # set color to an array/list of desired values
                          #colorscale='Viridis',   # choose a colorscale
                         # opacity=0.95
                      ),name=name))


            if stack:
                for s in range(Xqn.shape[0]):
                    if s == 4:
                        fig.add_trace(go.Surface(z=Zqn[s,:,:], x=Xqn[s,:,:], y=Yqn[s,:,:], surfacecolor = Vi[s,:,:],opacity=0.8,colorscale='jet',showscale=False,name = name  ))
                    else:
                        fig.add_trace(go.Surface(z=Zqn[s,:,:], x=Xqn[s,:,:], y=Yqn[s,:,:], surfacecolor = Vi[s,:,:],opacity=0.8,colorscale='jet',showscale=False))

            else:
                fig.add_trace(go.Surface(z=Zqn, x=Xqn, y=Yqn, surfacecolor = Vi,opacity=0.8,colorscale='Viridis',showscale=False,name = name ))

            if x2:
                if stack:
                    for s in range(Xqn.shape[0]):
                        if s == 4:
                            fig.add_trace(go.Surface(z=Zqn2[s,:,:], x=Xqn2[s,:,:], y=Yqn2[s,:,:], surfacecolor = Vi2[s,:,:],opacity=0.6,colorscale='gray',showscale=False,name = name  ))
                        else:
                            fig.add_trace(go.Surface(z=Zqn2[s,:,:], x=Xqn2[s,:,:], y=Yqn2[s,:,:], surfacecolor = Vi2[s,:,:],opacity=0.6,colorscale='gray',showscale=False))

                else:
                  fig.add_trace(go.Surface(z=Zqn2, x=Xqn2, y=Yqn2, surfacecolor = Vi2,opacity=0.8,colorscale='jet',showscale=False,name = name ))

            # fig.add_trace(go.Scatter3d(x = Xqn.flatten(), y = Yqn.flatten(), z = Zqn.flatten(),mode = 'markers',  marker=dict(
            #         size=3,
            #         color=Vi.flatten(),                # set color to an array/list of desired values
            #         colorscale='Viridis',   # choose a colorscale
            #         opacity=0.5
            #     )))
        fig.update_traces(showlegend=True)
        fig.update_scenes(zaxis_autorange="reversed")
        #fig.update_scenes(zaxis_showgrid= False,xaxis_showgrid= False,yaxis_showgrid= False)
        # if Volume_realsize is not None:
        #     fig.update_layout(scene_aspectratio=dict(x=Volume_realsize[0]*5, y=Volume_realsize[1]*5, z=Volume_realsize[2]*5))
        #fig.update_layout(scene_aspectratio=dict(x=1, y=2, z=2))
        path_fig = path_folder + 'volumes+planes_Patient_' +Result['Patients'][data_idx] + '.html'
        fig.write_html(path_fig)
        #fig.show()

        #return fig


def save_animation(VI_T,XQN_T,YQN_T,ZQN_T,Volume_realsize, list_iterator,
                label_params = None,
                save_path = "/Users/javierbisbal/Documents/GoogleDrive_notsync/Automatic_reformatting/Plotly_figs/Pruebas/prueba_animacion.html",
                vel=30,VI_lab=None,XQN_lab=None,YQN_lab=None,ZQN_lab=None, acts=None ):


    # Define frames

    nb_frames = VI_T.shape[0]
    def frame_args(duration):
      return {
              "frame": {"duration": duration},
              "mode": "immediate",
              "fromcurrent": True,
              "transition": {"duration": duration, "easing": "linear"},
          }

    if VI_lab is not None:
        fig = go.Figure(frames=[go.Frame(data=[go.Surface(
            z=np.squeeze(ZQN_T[k,:,:]),
            x=np.squeeze(XQN_T[k,:,:]),
            y=np.squeeze(YQN_T[k,:,:]),
            surfacecolor=np.squeeze(VI_T[k,:,:])
            #cmin=0, cmax=200
            ),go.Surface(
            z=ZQN_lab,
            x=XQN_lab,
            y=YQN_lab,
            surfacecolor=VI_lab,
            opacity = 0.5
            #cmin=0, cmax=200
            )],
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)])
    else:
        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            z=np.squeeze(ZQN_T[k,:,:]),
            x=np.squeeze(XQN_T[k,:,:]),
            y=np.squeeze(YQN_T[k,:,:]),
            surfacecolor=np.squeeze(VI_T[k,:,:]),
            #cmin=0, cmax=200
            ),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)])
    fig.update_layout(scene=dict(
                  xaxis=dict(visible=False),
                  yaxis=dict(visible=False),
                  zaxis=dict(visible=False),
                  )
                  )
    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=ZQN_T[0,:,:],
        x=XQN_T[0,:,:],
        y=YQN_T[0,:,:],
        surfacecolor=VI_T[0,:,:],
        colorscale='Gray',
        #cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    fig.add_trace(go.Surface(
        z=ZQN_lab,
        x=XQN_lab,
        y=YQN_lab,
        surfacecolor=VI_lab,
        colorscale='Gray',
        #cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    if acts is None:
        acts = np.zeros((list_iterator.shape[0],))
    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            #"label": str(round(change,1)),
                            # "label": 'step: '+ str(fr) + '. '+ str(round(list_iterator[fr,0],2))+ ' '+
                            # str(round(list_iterator[fr,1],2))+ ' '+
                            # str(round(list_iterator[fr,2],2))+ ' '+
                            # str(round(list_iterator[fr,3],5))+ ' '+
                            # str(round(list_iterator[fr,4],5)),

                            "label": 'step hola: '+ str(fr) + '. '+ str(round(list_iterator[fr,0],2))+ ' '+
                            str(round(np.cos(list_iterator[fr,1]),2))+ ' '+
                            str(round(np.cos(list_iterator[fr,2]),2))+ ' '+
                            str(round(list_iterator[fr,3],6)) + ' action: ' +
                            str(np.round(acts[fr],4)),

                            "method": "animate",
                        }
                        #for change, f in zip(list_iterator,fig.frames)
                        for fr, f in enumerate(fig.frames)
                    ],
                }
            ]
    # Layout
    print('SIZE ',Volume_realsize)
    fig.update_layout(
            title='Label Params (' + str(label_params) + ')',
            title_font_size = 12 ,
            width=800,
            height=800,
            scene=dict(#zaxis=dict(autorange="reversed"),
                        zaxis=dict(range=[-(Volume_realsize[2]/1.5), (Volume_realsize[2]/1.5)], autorange="reversed"),
                        yaxis=dict(range=[-(Volume_realsize[1]/1.5), (Volume_realsize[1]/1.5)]),
                        xaxis=dict(range=[-(Volume_realsize[0]/1.5), (Volume_realsize[0]/1.5)]),
                        #aspectratio=dict(x=Volume_realsize[sample,0]*7, y=Volume_realsize[sample,1]*7, z=Volume_realsize[sample,2]*7),
                        aspectratio=dict(x=Volume_realsize[0]*5, y=Volume_realsize[1]*5, z=Volume_realsize[2]*5),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            #"args": [None, frame_args(100)],
                            "args": [None, frame_args(vel)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    fig.write_html(save_path)

def save_images(VI_net, VI_lab,Patients,min_steps,save_path=None,mid_slice=4):
    from skimage.metrics import structural_similarity as ssim

    num_figures = VI_net.shape[0]
    print('Figuras:',num_figures)
    num_plot = num_figures
    idx_sample = 0
    MeanTotalSSim = np.zeros(VI_net.shape[0],)
    exit = False
    idx_figure = 1
    #mid_slice = VI_lab[0]['VI'].shape[0]//2
    while not exit:
        rows = min(4,max(num_plot//2,4))
        fig , ax = plt.subplots(rows,4,figsize=(14,8))
        ax_idx = 0

        for _ in range(len(ax.flat)//2):
            #print(np.squeeze(VI_net[idx_sample,min_steps[idx_sample],:,:]).dtype,np.squeeze(VI_lab[idx_sample]['VI'][4,:,:]).dtype)
            ssim_mean = ssim(np.squeeze(VI_net[idx_sample,min_steps[idx_sample],:,:]),np.squeeze(VI_lab[idx_sample]['VI'][mid_slice,:,:]))
            MeanTotalSSim[idx_sample] = ssim_mean
            ax.flat[ax_idx].imshow(np.squeeze(VI_net[idx_sample,min_steps[idx_sample],:,:]))
            ax.flat[ax_idx].set_title('Patient ' + Patients[idx_sample] + ' SSIM: ' + str(round(ssim_mean,3)))
            ax_idx+=1
            ax.flat[ax_idx].imshow(np.squeeze(VI_lab[idx_sample]['VI'][mid_slice,:,:]))
            ax.flat[ax_idx].set_title('Patient ' + Patients[idx_sample] + ' Label')
            num_plot -=1
            if num_plot  <= 0:
                break
            else:
                ax_idx+=1
                idx_sample+=1
        if num_plot > 0:
            idx_figure+=1
            fig.tight_layout()
            fig.savefig(save_path + '/images_'+ str(idx_figure)+'.jpeg')
            plt.close(fig)
        else:
            fig.tight_layout()
            fig.savefig(save_path + '/images_' + str(idx_figure)+ '.jpeg')
            plt.close(fig)
            exit = True

    return np.mean(MeanTotalSSim),np.std(MeanTotalSSim)

def save_graphs(TotalDistErrors,DistErrors,DistAngleErrors,AngleErrors,Rewards,Terminals,DistSteps,AngleSteps,Min_steps,Q_values=None,save_path=None):

    n_patients = TotalDistErrors.shape[0]
    F=16
    Mean_TotalDistErrors = np.mean(TotalDistErrors[range(n_patients),Min_steps])
    STD_TotalDistErrors = np.std(TotalDistErrors[range(n_patients),Min_steps])
    Mean_DistErrors = np.mean(DistErrors[range(n_patients),Min_steps])
    STD_DistErrors = np.std(DistErrors[range(n_patients),Min_steps])
    Mean_DistAngleErrors = np.mean(DistAngleErrors[range(n_patients),Min_steps])
    STD_DistAngleErrors = np.std(DistAngleErrors[range(n_patients),Min_steps])
    Mean_AngleErrors = np.mean(AngleErrors[range(n_patients),Min_steps])
    STD_AngleErrors = np.std(AngleErrors[range(n_patients),Min_steps])
    Mean_Rewards = np.mean(Rewards[range(n_patients),Min_steps])
    STD_Rewards = np.std(Rewards[range(n_patients),Min_steps])

    if Q_values is not None:
        Mean_Q_values = np.mean(Q_values[range(n_patients),Min_steps])
        STD_Q_values = np.std(Q_values[range(n_patients),Min_steps])

    fig , ax = plt.subplots(2,3,figsize=(40,20))


    ax[0][0].plot(np.mean(TotalDistErrors,axis=0))
    #ax[0][0].set_title('TotalDist mean: ' + str(round(Mean_TotalDistErrors,2)) +' std: ' + str(round(STD_TotalDistErrors,2)),fontsize=18)
    ax[0][0].set_title('Plane parameters euclidean distance',fontsize=18)
    ax[0][0].set_xlabel('step',fontsize=F)
    plt.setp(ax[0][0].get_xticklabels(), fontsize=F)
    plt.setp(ax[0][0].get_yticklabels(), fontsize=F)

    ax[0][1].plot(np.mean(DistErrors,axis=0))
    #ax[0][1].set_title('Dist mean: ' + str(round(Mean_DistErrors,2)) +' std: ' + str(round(STD_DistErrors,2)),fontsize=18)
    ax[0][1].set_title('Distance error',fontsize=18)
    ax[0][1].set_ylabel('mm',fontsize=F)
    ax[0][1].set_xlabel('step',fontsize=F)
    plt.setp(ax[0][1].get_xticklabels(), fontsize=F)
    plt.setp(ax[0][1].get_yticklabels(), fontsize=F)

    ax[0][2].plot(np.mean(AngleErrors,axis=0))
    #ax[0][2].set_title('Angle Errors mean: ' + str(round(Mean_AngleErrors,2)) +' std: ' + str(round(STD_AngleErrors,2)),fontsize=18)
    ax[0][2].set_title('Angle error',fontsize=18)
    ax[0][2].set_ylabel('degrees',fontsize=F)
    ax[0][2].set_xlabel('step',fontsize=F)
    plt.setp(ax[0][2].get_xticklabels(), fontsize=F)
    plt.setp(ax[0][2].get_yticklabels(), fontsize=F)


    ax[1][0].plot(np.mean(Rewards,axis=0))
    ax[1][0].set_title('Cumulative reward mean: ' + str(round(Mean_Rewards,2)) +' std: ' + str(round(STD_Rewards,2)),fontsize=18)
    ax[1][0].set_xlabel('step',fontsize=F)
    plt.setp(ax[1][0].get_xticklabels(), fontsize=F)
    plt.setp(ax[1][0].get_yticklabels(), fontsize=F)

    ax[1][1].plot(np.mean(Terminals,axis=0))
    ax[1][1].set_title('Terminals mean ')
    ax[1][1].set_xlabel('step',fontsize=F)
    plt.setp(ax[1][1].get_xticklabels(), fontsize=F)
    plt.setp(ax[1][1].get_yticklabels(), fontsize=F)

    # ax[1][2].plot(np.mean(DistSteps,axis=0))
    # ax[1][2].set_title('Dist step mean ')
    # ax[1][2].set_xlabel('step')

    # ax[1][3].plot(np.mean(AngleSteps,axis=0))
    # ax[1][3].set_title('Angle step mean ')
    # ax[1][3].set_xlabel('step')
    if Q_values is not None:
        ax[1][2].plot(np.mean(Q_values,axis=0))
        ax[1][2].set_title('Q values mean: ' + str(round(Mean_Q_values,2)) +' std: ' + str(round(STD_Q_values,2)))
        ax[1][2].set_ylabel('Q value')
        ax[1][2].set_xlabel('step')
        plt.setp(ax[1][2].get_xticklabels(), fontsize=F)
        plt.setp(ax[1][2].get_yticklabels(), fontsize=F)

    fig.savefig(save_path,dpi=400)
    plt.close(fig)

def save_hists(TotalDistErrors,DistErrors,DistAngleErrors,AngleErrors,Rewards,Terminals,DistSteps,AngleSteps,Min_steps,Patients=None,Q_values=None,save_path=None,DATASETS=False):


    sb.color_palette("tab10")
    # print(AngleErrors.shape,Min_steps.shape)
    # print(AngleErrors,Min_steps)
    n_patients = TotalDistErrors.shape[0]
    F = 16
    Mean_TotalDistErrors = np.mean(TotalDistErrors[range(n_patients),Min_steps])
    STD_TotalDistErrors = np.std(TotalDistErrors[range(n_patients),Min_steps])
    Mean_DistErrors = np.mean(DistErrors[range(n_patients),Min_steps])
    STD_DistErrors = np.std(DistErrors[range(n_patients),Min_steps])
    Mean_DistAngleErrors = np.mean(DistAngleErrors[range(n_patients),Min_steps])
    STD_DistAngleErrors = np.std(DistAngleErrors[range(n_patients),Min_steps])
    Mean_AngleErrors = np.mean(AngleErrors[range(n_patients),Min_steps])
    STD_AngleErrors = np.std(AngleErrors[range(n_patients),Min_steps])
    Mean_Rewards = np.mean(Rewards[range(n_patients),Min_steps])
    STD_Rewards = np.std(Rewards[range(n_patients),Min_steps])

    Min_TotalDistErrors = np.squeeze(TotalDistErrors[range(n_patients),Min_steps])
    Min_DistErrors = np.squeeze(DistErrors[range(n_patients),Min_steps])
    Min_DistAngleErrors = np.squeeze(DistAngleErrors[range(n_patients),Min_steps])
    Min_AngleErrors = np.squeeze(AngleErrors[range(n_patients),Min_steps])
    Min_Rewards = np.squeeze(Rewards[range(n_patients),Min_steps])
    Min_Terminals = np.squeeze(Terminals[range(n_patients),Min_steps])
    # print(Min_steps.shape)
    # print(TotalDistErrors.shape,Min_DistErrors.shape)
    # print(AngleErrors.shape,Min_AngleErrors.shape)
    # print(Min_AngleErrors,Min_DistErrors)

    datasets = ['PAR-REC','Fallot (Type 2)','Garcia Control','Coarcation','Fallot (Type 1)','GE Controles','BAV']
    #datasets = ['PAR-REC','Garcia']
    if Patients is not None:
        IPCMRA_Angle_error = []
        CD_Angle_error = []
        IPCMRA_Dist_error = []
        CD_Dist_error = []
        for i, patient in enumerate(Patients):
            if ((int(patient[:3])) >= 0 and int(patient[:3]) <= 20) or ((int(patient[:3])) >= 150 and int(patient[:3]) <= 160) or ((int(patient[:3])) >= 200 and int(patient[:3]) <= 220):
                IPCMRA_Angle_error.append(Min_AngleErrors[i])
                IPCMRA_Dist_error.append(Min_DistErrors[i])
            else:
                CD_Angle_error.append(Min_AngleErrors[i])
                CD_Dist_error.append(Min_DistErrors[i])

        Datasets_Angle_error = {}
        Datasets_Dist_error = {}
        for dataset in datasets:
            Datasets_Angle_error[dataset] = []
            Datasets_Dist_error[dataset] = []
        for i, patient in enumerate(Patients):
            if ((int(patient[:3])) >= 0 and int(patient[:3]) <= 20):
                Datasets_Angle_error[datasets[0]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[0]].append(Min_DistErrors[i])
            elif ((int(patient[:3])) >= 150 and int(patient[:3]) <= 160):
                Datasets_Angle_error[datasets[1]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[1]].append(Min_DistErrors[i])
            elif ((int(patient[:3])) >= 200 and int(patient[:3]) <= 220):
                Datasets_Angle_error[datasets[2]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[2]].append(Min_DistErrors[i])
            elif ((int(patient[:3])) >= 40 and int(patient[:3]) <= 60):
                Datasets_Angle_error[datasets[3]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[3]].append(Min_DistErrors[i])
            elif ((int(patient[:3])) >= 80 and int(patient[:3]) <= 100):
                Datasets_Angle_error[datasets[4]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[4]].append(Min_DistErrors[i])
            elif ((int(patient[:3])) >= 221 and int(patient[:3]) <= 260):
                Datasets_Angle_error[datasets[5]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[5]].append(Min_DistErrors[i])
            elif ((int(patient[:3])) >= 300 and int(patient[:3]) <= 315):
                Datasets_Angle_error[datasets[6]].append(Min_AngleErrors[i])
                Datasets_Dist_error[datasets[6]].append(Min_DistErrors[i])


    if Q_values is not None:
        Mean_Q_values = np.mean(Q_values[:,Min_steps])
        STD_Q_values = np.std(Q_values[:,Min_steps])

    fig , ax = plt.subplots(2,3,figsize=(40,20))

    sb.histplot(Min_TotalDistErrors,ax = ax[0][0],kde=True,bins=10)
    #ax[0][0].hist(Min_TotalDistErrors,bins=10,density=True)
    ax[0][0].set_title('TotalDist mean: ' + str(round(Mean_TotalDistErrors,2)) +' std: ' + str(round(STD_TotalDistErrors,2)),fontsize=18)
    ax[0][0].set_xlim([0, 3])
    plt.setp(ax[0][0].get_xticklabels(), fontsize=F)
    plt.setp(ax[0][0].get_yticklabels(), fontsize=F)

    sb.histplot(Min_DistErrors,ax=ax[0][1], kde=True,bins=10)
    #ax[0][1].hist(Min_DistErrors,bins=10,density=True)
    ax[0][1].set_title('Dist mean: ' + str(round(Mean_DistErrors,2)) +' std: ' + str(round(STD_DistErrors,2)),fontsize=18)
    ax[0][1].set_xlabel('Distance error (mm)',fontsize=F)
    ax[0][1].set_xlim([0, 35])
    plt.setp(ax[0][1].get_xticklabels(), fontsize=F)
    plt.setp(ax[0][1].get_yticklabels(), fontsize=F)

    if Patients is not None:
        if DATASETS:
            for dataset in datasets:
                ax[0][2].scatter(Datasets_Dist_error[dataset],Datasets_Angle_error[dataset],s=100,marker='*')
                ax[0][2].set_ylim([0, 80])
                ax[0][2].set_xlim([0, 35])
                # ax[0][2].set_ylim([0, 45])
                # ax[0][2].set_xlim([0, 20])
        else:
            ax[0][2].scatter(IPCMRA_Dist_error,IPCMRA_Angle_error,s=100,marker='*')
            ax[0][2].scatter(CD_Dist_error,CD_Angle_error,s=100,marker='*')
            # ax[0][2].set_ylim([0, 45])
            # ax[0][2].set_xlim([0, 20])
            ax[0][2].set_ylim([0, 90])
            ax[0][2].set_xlim([0, 30])

        #ax[0][1].hist(Min_DistErrors,bins=10,density=True)
        #ax[0][2].set_title('Dist mean: ' + str(round(Mean_DistErrors,2)) +' std: ' + str(round(STD_DistErrors,2)),fontsize=18)
        ax[0][2].set_xlabel('Distance error (mm)',fontsize=F)
        ax[0][2].set_ylabel('Angle error (degrees)',fontsize=F)
        plt.setp(ax[0][2].get_xticklabels(), fontsize=F)
        plt.setp(ax[0][2].get_yticklabels(), fontsize=F)
        if DATASETS:
            ax[0][2].legend(datasets)
        else:
            ax[0][2].legend(['IPCMRA','CD'])

    sb.histplot(Min_AngleErrors,ax=ax[1][1], kde=True,bins=10)
    #ax[1][0].hist(Min_AngleErrors,bins=10,density=True)
    ax[1][1].set_title('Angle Errors mean: ' + str(round(Mean_AngleErrors,2)) +' std: ' + str(round(STD_AngleErrors,2)),fontsize=18)
    ax[1][1].set_xlabel('Angle error (degrees)',fontsize=F)
    ax[1][1].set_xlim([0, 40])
    plt.setp(ax[1][1].get_xticklabels(), fontsize=F)
    plt.setp(ax[1][1].get_yticklabels(), fontsize=F)

    # if Patients is not None:
    #     if DATASETS:
    #         pass
    #         # data = pd.DataFrame(data = Datasets_Angle_error)
    #         # sb.histplot(data,ax=ax[1][2], kde=True,bins=10)
    #     else:
    #         pass
    #         # sb.histplot(IPCMRA_Angle_error,ax=ax[1][2], kde=True,bins=10)
    #         # sb.histplot(CD_Angle_error,ax=ax[1][2], kde=True,bins=10)
    #
    #     #ax[1][0].hist(Min_AngleErrors,bins=10,density=True)
    #     ax[1][2].set_title('Angle Errors mean: ' + str(round(Mean_AngleErrors,2)) +' std: ' + str(round(STD_AngleErrors,2)),fontsize=18)
    #     ax[1][2].set_xlabel('Angle error (degrees)',fontsize=F)
    #     ax[1][2].set_xlim([0, 40])
    #     plt.setp(ax[1][2].get_xticklabels(), fontsize=F)
    #     plt.setp(ax[1][2].get_yticklabels(), fontsize=F)
    #     if DATASETS:
    #         ax[1][2].legend(datasets)
    #     else:
    #         ax[1][2].legend(['IPCMRA','CD'])

    sb.histplot(Min_Rewards,ax=ax[1][0], kde=True,bins=10)
    #ax[1][1].hist(Min_Rewards,bins=10,density=True)
    ax[1][0].set_title('Cumulative reward mean: ' + str(round(Mean_Rewards,2)) +' std: ' + str(round(STD_Rewards,2)),fontsize=18)
    plt.setp(ax[1][0].get_xticklabels(), fontsize=F)
    plt.setp(ax[1][0].get_yticklabels(), fontsize=F)

    # sb.histplot(Min_Terminals,ax=ax[1][2])
    # #ax[1][2].hist(Min_Terminals,bins=10,density=True)
    # ax[1][2].set_title('Terminals mean ')

    # ax[1][2].hist(Min_DistSteps,bins=10,density=True)
    # ax[1][2].set_title('Dist step mean ')
    # ax[1][2].set_xlabel('step')

    # ax[1][3].plot(np.mean(AngleSteps,axis=0))
    # ax[1][3].set_title('Angle step mean ')
    # ax[1][3].set_xlabel('step')
    if Q_values is not None:
        ax[1][2].plot(np.mean(Q_values,axis=0))
        ax[1][2].set_title('Q values mean: ' + str(round(Mean_Q_values,2)) +' std: ' + str(round(STD_Q_values,2)))
        ax[1][2].set_ylabel('Q value')
        ax[1][2].set_xlabel('step')
        plt.setp(ax[1][2].get_xticklabels(), fontsize=F)
        plt.setp(ax[1][2].get_yticklabels(), fontsize=F)

    fig.savefig(save_path,dpi=400)
    plt.close(fig)

def save_graphs_full(TotalDistErrors,DistErrors,DistAngleErrors,AngleErrors,Rewards,
                    Terminals,DistSteps,AngleSteps,Patients,
                    Q_values=None,
                    save_path=None,
                    Values=None,
                    Advantages=None,
                    Entropies=None):

    F =16
    for idx,patient in enumerate(Patients):
        fig , ax = plt.subplots(2,4,figsize=(40,20))

        ax[0][0].plot(TotalDistErrors[idx,:])
        ax[0][0].set_title('TotalDist mean: ' + str(round(np.mean(TotalDistErrors[idx,:],axis=0),2)),fontsize=18)
        ax[0][0].set_xlabel('step',fontsize=F)
        plt.setp(ax[0][0].get_xticklabels(), fontsize=F)
        plt.setp(ax[0][0].get_yticklabels(), fontsize=F)

        ax[0][1].plot(DistErrors[idx,:])
        ax[0][1].set_title('Dist mean: ' + str(round(np.mean(DistErrors[idx,:],axis=0),2)),fontsize=18)
        ax[0][1].set_ylabel('Distance error (mm)',fontsize=F)
        ax[0][1].set_xlabel('step',fontsize=F)
        plt.setp(ax[0][1].get_xticklabels(), fontsize=F)
        plt.setp(ax[0][1].get_yticklabels(), fontsize=F)

        ax[0][2].plot(DistAngleErrors[idx,:])
        ax[0][2].set_title('Dist Angle mean: ' + str(round(np.mean(DistAngleErrors[idx,:],axis=0),2)),fontsize=18)
        ax[0][2].set_xlabel('step',fontsize=F)
        plt.setp(ax[0][2].get_xticklabels(), fontsize=F)
        plt.setp(ax[0][2].get_yticklabels(), fontsize=F)

        ax[0][3].plot(AngleErrors[idx,:])
        ax[0][3].set_title('Angle Errors mean: ' + str(round(np.mean(AngleErrors[idx,:],axis=0),2)),fontsize=18)
        ax[0][3].set_ylabel('Angle error (degrees)',fontsize=F)
        ax[0][3].set_xlabel('step',fontsize=F)
        plt.setp(ax[0][3].get_xticklabels(), fontsize=F)
        plt.setp(ax[0][3].get_yticklabels(), fontsize=F)

        ax[1][0].plot(Rewards[idx,:])
        ax[1][0].set_title('Cumulative reward mean: ' + str(round(np.mean(Rewards[idx,:],axis=0),2)),fontsize=18)
        ax[1][0].set_xlabel('step',fontsize=F)
        plt.setp(ax[1][0].get_xticklabels(), fontsize=F)
        plt.setp(ax[1][0].get_yticklabels(), fontsize=F)

        if Values is not None:
            ax[1][1].plot(Values[idx,:])
            ax[1][1].set_title('Values mean: ' + str(round(np.mean(Values[idx,:],axis=0),2)),fontsize=18)
            ax[1][1].set_xlabel('step',fontsize=F)
            plt.setp(ax[1][1].get_xticklabels(), fontsize=F)
            plt.setp(ax[1][1].get_yticklabels(), fontsize=F)

        else:
            ax[1][1].plot(Terminals[idx,:])
            ax[1][1].set_title('Terminals mean: ' + str(round(np.mean(Terminals[idx,:],axis=0),2)),fontsize=18)
            ax[1][1].set_xlabel('step',fontsize=F)
            plt.setp(ax[1][1].get_xticklabels(), fontsize=F)
            plt.setp(ax[1][1].get_yticklabels(), fontsize=F)

        if Advantages is not None:
            ax[1][2].plot(Advantages[idx,:])
            ax[1][2].set_title('Advantages ' + str(round(np.mean(Advantages[idx,:],axis=0),2)),fontsize=15)
            ax[1][2].set_xlabel('step',fontsize=F)
            plt.setp(ax[1][2].get_xticklabels(), fontsize=F)
            plt.setp(ax[1][2].get_yticklabels(), fontsize=F)
        else:
            ax[1][2].plot(DistSteps[idx,:])
            ax[1][2].set_title('Dist step mean: ' + str(round(np.mean(DistSteps[idx,:],axis=0),2)),fontsize=15)
            ax[1][2].set_xlabel('step',fontsize=F)
            plt.setp(ax[1][2].get_xticklabels(), fontsize=F)
            plt.setp(ax[1][2].get_yticklabels(), fontsize=F)


        if Q_values is not None:
            ax[1][3].plot(Q_values[idx,:])
            ax[1][3].set_title('Q_values mean: ' + str(round(np.mean(Q_values[idx,:],axis=0),2)),fontsize=15)
            ax[1][3].set_xlabel('step',fontsize=F)
            plt.setp(ax[1][3].get_xticklabels(), fontsize=F)
            plt.setp(ax[1][3].get_yticklabels(), fontsize=F)
        if Entropies is not None:
            for act in range(4):
                ax[1][3].plot(np.squeeze(Entropies[idx,:,act]))
            ax[1][3].set_title('Entropies',fontsize=15)
            ax[1][3].set_xlabel('step',fontsize=F)
            plt.setp(ax[1][3].get_xticklabels(), fontsize=F)
            plt.setp(ax[1][3].get_yticklabels(), fontsize=F)



        fig.savefig(save_path + 'Patient_' + patient + '.png')
        plt.close(fig)

def info_visualizer(info_training,info_valid, save_path,info_training2=None,info_valid2=None,legend = None):

    fig, ax = plt.subplots(2,3,figsize=(14,8))

    steps = np.arange(0,info_training['Mean_episodes_rewards'].shape[0]*1000,1000)

    ax.flat[0].plot(steps,info_training['Mean_episodes_rewards'])
    if info_training2 is not None:
        ax.flat[0].plot(steps,info_training2['Mean_episodes_rewards'])
    ax.flat[0].set_title('Moving average 100 episodes rewards, Best: ' + str(round(info_training['best_mean_episode_reward'],2)))
    ax.flat[0].set_xlabel('steps')
    ax.flat[0].set_ylabel('moving average reward episodes ')
    ax.flat[0].set_ylim(bottom=-20)
    if legend is not None:
        ax.flat[0].legend(legend)

    TotalDistErrors_mean = []
    DistErrors_mean = []
    AngleErrors_mean = []
    Rewards_mean = []
    for elem in info_valid:
        TotalDistErrors_mean.append(elem['TotalDistErrors_mean'].item())
        DistErrors_mean.append(elem['DistErrors_mean'].item())
        AngleErrors_mean.append(elem['AngleErrors_mean'].item())
        Rewards_mean.append(elem['Rewards_mean'].item())

    if info_valid2 is not None:
        TotalDistErrors_mean2 = []
        DistErrors_mean2 = []
        AngleErrors_mean2 = []
        Rewards_mean2 = []
        for elem in info_valid2:
            TotalDistErrors_mean2.append(elem['TotalDistErrors_mean'].item())
            DistErrors_mean2.append(elem['DistErrors_mean'].item())
            AngleErrors_mean2.append(elem['AngleErrors_mean'].item())
            Rewards_mean2.append(elem['Rewards_mean'].item())

    steps_1e6 = np.arange(0,len(TotalDistErrors_mean)*1e5,1e5)
    ax.flat[1].plot(steps_1e6,TotalDistErrors_mean)
    if info_valid2 is not None:
        ax.flat[1].plot(steps_1e6,TotalDistErrors_mean2)
    ax.flat[1].set_title('TotalDistErrors_mean')
    ax.flat[1].set_xlabel('steps')
    ax.flat[1].set_ylim(bottom=0)
    if legend is not None:
        ax.flat[1].legend(legend)

    ax.flat[2].plot(steps_1e6,DistErrors_mean)
    if info_valid2 is not None:
        ax.flat[2].plot(steps_1e6,DistErrors_mean2)
    ax.flat[2].set_title('DistErrors_mean (mm)')
    ax.flat[2].set_xlabel('steps')
    ax.flat[2].set_ylim(bottom=0)
    if legend is not None:
        ax.flat[2].legend(legend)

    ax.flat[3].plot(steps_1e6,AngleErrors_mean)
    if info_valid2 is not None:
        ax.flat[3].plot(steps_1e6,AngleErrors_mean2)
    ax.flat[3].set_title('AngleErrors_mean (degrees)')
    ax.flat[3].set_xlabel('steps')
    ax.flat[3].set_ylim(bottom=0)
    if legend is not None:
        ax.flat[3].legend(legend)

    ax.flat[4].plot(steps_1e6,Rewards_mean)
    if info_valid2 is not None:
        ax.flat[4].plot(steps_1e6,Rewards_mean2)
    ax.flat[4].set_title('Rewards_mean')
    ax.flat[4].set_xlabel('steps')
    ax.flat[4].set_ylim(bottom=-50)
    if legend is not None:
        ax.flat[4].legend(legend)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def a3c_info_visualizer(info_valid, save_path,info_valid2=None,legend = None):

    fig, ax = plt.subplots(2,2,figsize=(14,8))

    TotalDistErrors = np.array(info_valid['TotalDistError_L'])
    DistErrors = np.array(info_valid['OriginsError_L'])
    AngleErrors = np.array(info_valid['AngleError_L'])
    Rewards = np.array(info_valid['reward_total_L'])

    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w), 'same') / w
    #
    # window = 50
    # TotalDistErrors = moving_average(TotalDistErrors,window)
    # DistErrors = moving_average(DistErrors,window)
    # AngleErrors = moving_average(AngleErrors,window)
    # Rewards = moving_average(Rewards,window)


    if info_valid2 is not None:
        TotalDistErrors2 = info_valid2['TotalDistError_L']
        DistErrors2 = info_valid2['AngleError_L']
        AngleErrors2 = info_valid2['OriginsError_L']
        Rewards2 = info_valid2['reward_total_L']
        TotalDistErrors2 = moving_average(TotalDistErrors2,window)
        DistErrors2 = moving_average(DistErrors2,window)
        AngleErrors2 = moving_average(AngleErrors2,window)
        Rewards2 = moving_average(Rewards2,window)


    steps = np.arange(0,len(info_valid['TotalDistError_L']))

    ax.flat[0].plot(steps,Rewards)
    if info_valid2 is not None:
        ax.flat[0].plot(steps,Rewards2)
    ax.flat[0].set_title('Average episodes rewards')
    ax.flat[0].set_xlabel('steps')
    ax.flat[0].set_ylabel('Average reward episodes ')
    if legend is not None:
        ax.flat[0].legend(legend)
    ax.flat[0].set_ylim(bottom=-15)

    ax.flat[1].plot(steps,TotalDistErrors)
    if info_valid2 is not None:
        ax.flat[1].plot(steps_1e6,TotalDistErrors)
    ax.flat[1].set_title('TotalDistErrors')
    ax.flat[1].set_xlabel('steps')
    ax.flat[1].set_ylim(bottom=0)

    if legend is not None:
        ax.flat[1].legend(legend)

    ax.flat[2].plot(steps,DistErrors)
    if info_valid2 is not None:
        ax.flat[2].plot(steps_1e6,DistErrors)
    ax.flat[2].set_title('DistErrors (mm)')
    ax.flat[2].set_xlabel('steps')
    ax.flat[2].set_ylim(bottom=0)
    if legend is not None:
        ax.flat[2].legend(legend)

    ax.flat[3].plot(steps,AngleErrors)
    if info_valid2 is not None:
        ax.flat[3].plot(steps_1e6,AngleErrors)
    ax.flat[3].set_title('AngleErrors (degrees)')
    ax.flat[3].set_xlabel('steps')
    ax.flat[3].set_ylim(bottom=0)
    if legend is not None:
        ax.flat[3].legend(legend)
    #
    # ax.flat[4].plot(steps_1e6,Rewards_mean)
    # if info_valid2 is not None:
    #     ax.flat[4].plot(steps_1e6,Rewards_mean2)
    # ax.flat[4].set_title('Rewards_mean')
    # ax.flat[4].set_xlabel('steps')
    # if legend is not None:
    #     ax.flat[4].legend(legend)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def Concat_Results(Results1,Results2):

    Results = {'XQN_T':np.concatenate(Results1['XQN_T'],Results2['XQN_T'],axis=0),
        'YQN_T':np.concatenate(Results1['YQN_T'],Results2['YQN_T'],axis=0),
        'ZQN_T':np.concatenate(Results1['ZQN_T'],Results2['ZQN_T'],axis=0),
        'VI_T':np.concatenate(Results1['VI_T'],Results2['VI_T'],axis=0),
        'Patients':Results1['Patients'] + Results2['Patients'],
        'Min_steps':np.concatenate(Results1['Min_steps'],Results2['Min_steps'],axis=0),
        'TotalDistErrors':np.concatenate(Results1['TotalDistErrors'],Results2['TotalDistErrors'],axis=0),
        'DistErrors':np.concatenate(Results1['DistErrors'],Results2['DistErrors'],axis=0),
        'AngleErrors':np.concatenate(Results1['AngleErrors'],Results2['AngleErrors'],axis=0),
        'DistAngleErrors':np.concatenate(Results1['DistAngleErrors'],Results2['DistAngleErrors'],axis=0),
        'DistSteps':np.concatenate(Results1['DistSteps'],Results2['DistSteps'],axis=0),
        'AngleSteps':np.concatenate(Results1['AngleSteps'],Results2['AngleSteps'],axis=0),
        'Rewards':np.concatenate(Results1['Rewards'],Results2['Rewards'],axis=0),
        'Terminals':np.concatenate(Results1['Terminals'],Results2['Terminals'],axis=0),
        'Planes_params':np.concatenate(Results1['Planes_params'],Results2['Planes_params'],axis=0),
        'Planes_lab_params':np.concatenate(Results1['Planes_lab_params'],Results2['Planes_lab_params'],axis=0),
        'plane_labs':Results1['plane_labs'] + Results2['plane_labs'],
        'Volumes':Results1['Volumes'] + Results2['Volumes'],
        'Volumes_realsize':np.concatenate(Results1['Volumes_realsize'],Results2['Volumes_realsize'],axis=0),
        'OriginsErrors':np.concatenate(Results1['OriginsErrors'],Results2['OriginsErrors'],axis=0),
        'Origins':np.concatenate(Results1['Origins'],Results2['Origins'],axis=0),
        'Origins_lab':np.concatenate(Results1['Origins_lab'],Results2['Origins_lab'],axis=0),
        'Actions':np.concatenate(Results1['Actions'],Results2['Actions'],axis=0),
        'SEGS':Results1['SEGS'] + Results2['SEGS']}

    return  Results
