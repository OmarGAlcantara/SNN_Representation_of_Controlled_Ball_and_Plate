
#This code simulates the dynamics of the ball and plate with a feedback state
#control using spiking neural networks
# Omar Alejandro García Alcántara  Cinvestav, Zacatenco.
# Diego Said Chavez Arana  NMSU, NM.

import matplotlib.pyplot as plt
import nengo
import numpy as nu

plt.rcParams['font.family'] = 'serif'  # Use a generic serif font
plt.rcParams['font.serif'] = 'DejaVu Serif'
plt.rcParams['pdf.fonttype'] = 42

gravity = 9.81
kp_x = -1.143
kd_x = -1.2
kp_y = -1.143
kd_y = -1.25

t_synapse = 0.2

r = 0.1;
N = 2;
a = 0.5; 

with nengo.Network(label="ballandplate", seed = 2) as model:
    
#In order to model the system of the form dot{x} = Ax + Bu , this one is transformed to 
# dot{x}= 1/T(f'+g'-x) with f'=TAx + x and g'= TBu  al tiempo discreto

#Two ensembles are created to represent the matrices A and B
    
    f = nengo.Ensemble(n_neurons=600, dimensions=4, radius = 0.5)
    g = nengo.Ensemble(n_neurons=200, dimensions=2, radius = 0.5)
    
#The functions A_fun and B_fun model the terms f'=TAx and g'=TBu. 

    def f_fun(x):
        return [x[1]*t_synapse,0,x[3]*t_synapse,0]+x
    def g_fun(x):
        return [0,-nu.sin(x[0])*5/7*gravity*t_synapse,0,-nu.sin(x[1])*5/7*gravity*t_synapse]
    
    nengo.Connection(g,f,synapse = t_synapse,function=g_fun)
    nengo.Connection(f,f,synapse = t_synapse,function=f_fun)

#The signals stim_ref_x and stim_ref_y are the position references that be encoded through
#the ensemble ref
    

    Reference_Signals = nengo.Node(lambda t: [r*nu.sin(0.25*t),r*nu.cos(0.25*t)])  
    Encoded_Reference = nengo.Ensemble(n_neurons=300,dimensions=2, neuron_type = nengo.Direct())
    nengo.Connection(Reference_Signals,Encoded_Reference)

# the position of the ball that is sent to the error ensemble
#applying a transformation of -1. The error ensemble also receives the references signals to calculate
#the position error.
    
    Error = nengo.Ensemble(n_neurons=500,dimensions=2, radius = 0.15)
    err_syn = 0.005
    nengo.Connection(Encoded_Reference,Error, synapse = err_syn)
    nengo.Connection(f[[0,2]],Error,transform=-1, synapse = err_syn)
    

#The error derivatives are calculated using two ensembles where the signals are
#sending with two different synaptic delays and applying a transformation inversely
#proportional to the difference between the synaptic delays
    
    d_e_x = nengo.Ensemble(n_neurons=100, dimensions = 1, radius = 0.2)
    nengo.Connection(Error[0],d_e_x,synapse=0.05,transform=20)
    nengo.Connection(Error[0],d_e_x,synapse=0.1,transform=-20)
    
    d_e_y = nengo.Ensemble(n_neurons=100,dimensions=1, radius = 0.4)
    nengo.Connection(Error[1],d_e_y,synapse=0.05,transform=20)
    nengo.Connection(Error[1],d_e_y,synapse=0.1,transform=-20)
    
#The error and their derivatives are sent to the ensemble g applying the 
#transform corresponding to the control gains to generate the control signals
   
    def control(x):
        return [x[0]*kp_x,x[1]*kp_y]    
    
    controlEns = nengo.Ensemble(n_neurons= 800, dimensions=2)
    nengo.Connection(Error,controlEns,function=control)
    nengo.Connection(d_e_x,controlEns[0],transform=kd_x)
    nengo.Connection(d_e_y,controlEns[1],transform=kd_y)    
    nengo.Connection(controlEns, g, synapse=None, function=None)
    
    
    ens_probe = nengo.Probe(f,synapse=0.1)
    ref_probe = nengo.Probe(Reference_Signals,synapse=0.1)
    error_probe = nengo.Probe(Error,synapse=0.1)
    dex_probe = nengo.Probe(d_e_x,synapse=0.1)
    dey_probe = nengo.Probe(d_e_y,synapse=0.1)
    ref_Ens_probe = nengo.Probe(Encoded_Reference,synapse=0.1)
    #Pos_x_and_y_probe = nengo.Probe(Pos_x_and_y,synapse=0.1)
    g_probe = nengo.Probe(g, synapse = 0.1)
    controlPlot_probe = nengo.Probe(controlEns, synapse = 0.1)
    
    
    with nengo.Simulator(model) as sim:
        sim.run(30)
    t = sim.trange()


  
####################################################################################################################

    def plot_xy(data, num_xticks=5, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
        
        plt.figure(figsize=(9, 10))
        plt.plot(t, data[error_probe][:, 0], label="Position Error $x_{p}$ axis", color='blue')
        plt.plot(t, data[error_probe][:, 1], label="Position Error $y_{p}$ axis", color='red')
        plt.xlabel("Time [s]", fontsize=32)
        plt.ylabel("Position Errors [m]", fontsize=32)
        plt.grid(True)
        plt.legend(fontsize=32) 
    
        # Set the number of ticks on the x-axis and y-axis
        plt.locator_params(axis='x', nbins=num_xticks)
        plt.locator_params(axis='y', nbins=num_yticks)
    
        # Set the x and y axis limits if provided
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        
        file = '/home/omarg/control_quad_ws/src/nengo_examples/CCE 2023/ImagesCCE/' + 'Errors'
        plt.savefig(file, format="pdf", bbox_inches='tight')   
    
    plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.04, ymax=0.07)

###############################################################################################
 
    def plot_xy(data, num_xticks=5, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
        
        plt.figure(figsize=(9, 10))
        plt.plot(t, data[ref_probe][:,0], label="Reference $x_{d1}$", linewidth=4, color='blue')
        plt.plot(t, data[ens_probe][:,0], label="Position $x_{p}$ axis", color='red')
        plt.xlabel("Time [s]", fontsize=32)
        plt.ylabel("Position $x_{p}$ [m]", fontsize=32)
        plt.grid(True)
        plt.legend(fontsize=32, loc='upper right') 
    
        # Set the number of ticks on the x-axis and y-axis
        plt.locator_params(axis='x', nbins=num_xticks)
        plt.locator_params(axis='y', nbins=num_yticks)
     
        # Set the x and y axis limits if provided
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        
        file = '/home/omarg/control_quad_ws/src/nengo_examples/CCE 2023/ImagesCCE/' + 'Posx'
        plt.savefig(file, format="pdf", bbox_inches='tight')   
    
    plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.11, ymax=0.19)    
    
#######################################################################################################
 
    def plot_xy(data, num_xticks=5, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
        
        plt.figure(figsize=(9, 10))
        plt.plot(t, data[ref_probe][:,1], label="Reference $x_{d2}$", linewidth=4, color='blue')
        plt.plot(t, data[ens_probe][:,2], label="Position $y_{p}$ axis", color='red')
        plt.xlabel("Time [s]", fontsize=32)
        plt.ylabel("Position $y_{p}$ [m]", fontsize=32)
        plt.grid(True)
        plt.legend(fontsize=32, loc='upper right') 
    
        # Set the number of ticks on the x-axis and y-axis
        plt.locator_params(axis='x', nbins=num_xticks)
        plt.locator_params(axis='y', nbins=num_yticks)
    
        # Set the x and y axis limits if provided
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28) 
        
        file = '/home/omarg/control_quad_ws/src/nengo_examples/CCE 2023/ImagesCCE/' + 'Posy'
        plt.savefig(file, format="pdf", bbox_inches='tight')   
    
    plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.11, ymax=0.22)
    
 #######################################################################################################
    
    def plot_xy(data, num_xticks=4, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
        
        plt.figure(figsize=(9, 10))
        plt.plot(data[ref_probe][:, 0], data[ref_probe][:, 1], label="Reference", linewidth=4, color='blue')
        plt.plot(data[ens_probe][:, 0], data[ens_probe][:, 2], label="Position of the ball", color='red')
        plt.xlabel(r'Position $x_{p}$ [m]', fontsize=32)
        plt.ylabel(r'Position $y_{p}$ [m]', fontsize=32)
        plt.grid(True)
        plt.legend(fontsize=32, loc='lower right') 
    
        # Set the number of ticks on the x-axis and y-axis
        plt.locator_params(axis='x', nbins=num_xticks)
        plt.locator_params(axis='y', nbins=num_yticks)
    
        # Set the x and y axis limits if provided
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
         
        file = '/home/omarg/control_quad_ws/src/nengo_examples/CCE 2023/ImagesCCE/' + 'Circle'
        plt.savefig(file, format="pdf", bbox_inches='tight')    
    
    plot_xy(sim.data, num_xticks=4, num_yticks=6, xmin=-0.15, xmax=0.15, ymin=-0.17, ymax=0.15)
    
    
    ###################################################################################################
    
    def plot_xy(data, num_xticks=5, num_yticks=5, xmin=None, xmax=None, ymin=None, ymax=None):
        
        plt.figure(figsize=(20, 5))
        plt.plot(t, data[controlPlot_probe][:, 0], label="$u_1$", color='blue')
        plt.plot(t, data[controlPlot_probe][:, 1], label="$u_2$", color='red')
        plt.xlabel("Time [s]", fontsize=32)
        plt.ylabel("Control signals ", fontsize=32)
        plt.grid(True)
        plt.legend(fontsize=32) 
    
        # Set the number of ticks on the x-axis and y-axis
        plt.locator_params(axis='x', nbins=num_xticks)
        plt.locator_params(axis='y', nbins=num_yticks)
    
        # Set the x and y axis limits if provided
        if xmin is not None and xmax is not None:
            plt.xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        
        file = '/home/omarg/control_quad_ws/src/nengo_examples/CCE 2023/ImagesCCE/' + 'Control Sygnals'
        plt.savefig(file, format="pdf", bbox_inches='tight')   
    
    plot_xy(sim.data, num_xticks=6, num_yticks=6, xmin=-1, xmax=31, ymin=-0.45, ymax=0.45)
    
 
    
    