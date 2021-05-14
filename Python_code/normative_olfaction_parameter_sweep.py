import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
import normative_olfaction
from normative_olfaction import weighting_Matrix, get_optim_params, olfactory_model, postprocessing

M = 2 #Dimension of latent space
N = 41 #Dimension of neural state space
DRIFT = -0.25

# Global parameters
DT = 0.01 #Time interval
T_END = 10 #Total time for experiment
T_INIT = 1 #Initial no stimulus period
T_ON = 4 #Stimulus presence period
T_OFF = 5 #Stimulus withdrawal period

ERROR_PENALTY = 10 #default penalty for error in latent state
ENERGY_PENALTY = 4 #default penalty for energy expenditure
SMOOTHNESS_PENALTY = 0.2 #default penalty for large fluctuations
C = 11 #default overlap number

def run_prog(C = C, q = ERROR_PENALTY, s = ENERGY_PENALTY, r = SMOOTHNESS_PENALTY):
    b_Matrix = weighting_Matrix(N, C)

    #create an instance of the olfactory model
    create_olfactory_model = olfactory_model(DRIFT, b_Matrix, q, s, r)

    z_stim = np.zeros((M,int(T_END/DT)))
    z_stim[0, int(T_INIT/DT):int((T_INIT+T_ON)/DT)] = 1

    nu_init = np.zeros((M, 1))
    x_init = np.zeros((N,1))

    #forward simulate the olfactory_model
    x_Response, nu_Response = create_olfactory_model.forward_sim(z_stim, nu_init, x_init)

    #Post processing for plotting data
    x_blue_mean, x_blue_std, x_untuned_mean, x_untuned_std, x_red_mean, x_red_std = postprocessing(x_Response, C)
    t_index = np.arange(0,T_END,DT)

    return t_index, z_stim, x_blue_mean, x_untuned_mean, x_red_mean, nu_Response

# Create the widget for changing overlap between tuning curves
def sweep_overlap(C0 = C, delta_C = 5):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 12))
    plt.subplots_adjust(left = 0.25, bottom = 0.25)
    t_index, z_stim, x_blue_mean, x_untuned_mean, x_red_mean, nu_Response = run_prog(C = C0)

    l1, = ax1.plot(t_index, x_blue_mean, color='blue', label = 'Blue tuned neurons' )
    l2, = ax1.plot(t_index, x_untuned_mean, color='magenta', label = 'Untuned neurons')
    l3, = ax1.plot(t_index, x_red_mean, color='red', label ='Red tuned neurons' )
    ax1.set(xlabel='Time in seconds', ylabel='Mean neural activity')
    ax1.legend(loc = 'upper right')
    ax1.grid()
    ax1.margins(x=0)

    l4, = ax2.plot(nu_Response[0,:], nu_Response[1,:], color='blue')
    ax2.plot(z_stim[0,int(T_INIT/DT)], z_stim[1, int(T_INIT/DT)], marker='o', color='black')
    ax2.text(0.8, -0.15, 'Nominal Representation', fontsize=12)
    ax2.set_xlim([-0.2, 1.2])
    ax2.set_ylim([-0.2, 1.2])
    ax2.set(xlabel='Latent Dimension 1', ylabel='Latent Dimension 2')
    ax2.grid()

    axcolor = 'lightgoldenrodyellow'
    axC = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    C_sweep = Slider(axC, 'Overlapping Neurons', 1, N-1, valinit = C0, valstep = delta_C)

    def update(val):
        C  = C_sweep.val
        t_index, z_stim, x_blue_mean, x_untuned_mean, x_red_mean, nu_Response = run_prog(C)
        l1.set_ydata(x_blue_mean)
        l2.set_ydata(x_untuned_mean)
        l3.set_ydata(x_red_mean)

        l4.set_xdata(nu_Response[0,:])
        l4.set_ydata(nu_Response[1,:])
        fig.canvas.draw_idle()

    C_sweep.on_changed(update)
    plt.show()

#create a widget for sweeping over error penalty, energy penalty and smoothness penalty

def get_accuracy(t_index, z_stim, nu_Response):

    diff = z_stim[:,int((T_INIT+T_ON)/DT)-1] - nu_Response[:, int((T_INIT+T_ON)/DT)-1]
    accuracy = 1 - np.linalg.norm(diff)

    latency = -1
    for i in range(int(T_INIT/DT)-1, int((T_INIT+T_ON)/DT)):
        dist = np.linalg.norm(z_stim[:,int((T_INIT+T_ON)/DT)-1] - nu_Response[:, i])
        if dist<0.2:
            latency = t_index[i] - T_INIT
            break

    if latency<0:
        latency = (T_INIT+T_ON)/DT

    return accuracy, latency

def sweep_penalty(q0 = ERROR_PENALTY, delta_q = 2, s0 = ENERGY_PENALTY, delta_s = 2, r0 = SMOOTHNESS_PENALTY, delta_r = 0.1):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 16))
    plt.subplots_adjust(left = 0.25, bottom = 0.25)
    t_index, z_stim, x_blue_mean, x_untuned_mean, x_red_mean, nu_Response = run_prog(q = q0, s = s0, r = r0)
    accuracy, latency = get_accuracy(t_index, z_stim, nu_Response)

    l1, = ax1.plot(t_index, x_blue_mean, color='blue', label = 'Blue tuned neurons' )
    l2, = ax1.plot(t_index, x_untuned_mean, color='magenta', label = 'Untuned neurons')
    l3, = ax1.plot(t_index, x_red_mean, color='red', label ='Red tuned neurons' )
    ax1.set(xlabel='Time in seconds', ylabel='Mean neural activity')
    ax1.legend(loc = 'upper right')
    ax1.grid()
    ax1.margins(x=0)

    labels = ['Accuracy']
    w = 0.35
    pos = np.arange(len(labels))
    ax = ax2.twinx()
    l4 = ax2.bar(pos-w, [accuracy], width = w, label = 'Accuracy', color = 'pink')
    ax2.set(ylabel = 'Accuracy' )
    # ax2.legend(loc = 'lower left')
    ax2.set_ylim([0, 1.05])

    l5 = ax.bar(pos+w, [latency], width = w, label = 'Latency', color = 'gray')
    # ax.legend(loc = 'lower right')
    ax.set(ylabel = 'Time in secs')
    ax.set_ylim([0, 4])


    axcolor = 'lightgoldenrodyellow'
    axQ = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axS = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axR = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

    q_sweep = Slider(axQ, 'Penalty on error in latent representation', 2, 30, valinit = q0, valstep = delta_q)
    s_sweep = Slider(axS, 'Penalty on energy expenditure', 2, 30, valinit = s0, valstep = delta_s)
    r_sweep = Slider(axR, 'Penalty on rapid fluctuations', 0.1, 5, valinit = r0, valstep = delta_r)

    def update(val):
        q  = q_sweep.val
        s = s_sweep.val
        r = r_sweep.val
        t_index, z_stim, x_blue_mean, x_untuned_mean, x_red_mean, nu_Response = run_prog(q = q, s = s)
        accuracy, latency = get_accuracy(t_index, z_stim, nu_Response)

        l1.set_ydata(x_blue_mean)
        l2.set_ydata(x_untuned_mean)
        l3.set_ydata(x_red_mean)

        ax2.clear()
        ax.clear()
        l4 = ax2.bar(pos - w, [accuracy], width = 0.35, label = 'Accuracy', color = 'pink')
        # ax2.legend(loc = 'lower left')
        ax2.set(ylabel = 'Accuracy' )
        ax2.set_ylim([0, 1.05])

        l5 = ax.bar(pos+w, [latency], width = w, label = 'Latency', color = 'gray')
        # ax.legend(loc = 'lower right')
        ax.set(ylabel = 'Time in secs')
        ax.set_ylim([0, 4])

        fig.canvas.draw_idle()

    q_sweep.on_changed(update)
    s_sweep.on_changed(update)
    r_sweep.on_changed(update)
    plt.show()

# Run Widget for sweeping amount of overlap
# sweep_overlap()

#Run Widget for sweeping over penalty
# sweep_penalty()
