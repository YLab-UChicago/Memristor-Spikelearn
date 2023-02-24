from pdb import set_trace
import numpy as np
from sys import argv,path
import itertools
import math
import os

path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))

from spikelearn import SpikingNet, HomeostasisLayer, SecondOrderLayer, PlasticSynapse, LambdaSynapse, OneToOneSynapse, SynapseCircuit, cell_Paiyu_chen_15, BaseSynapse, cell_Vteam, cell_data_driven
from spikelearn.generators import Poisson

tstep_logical=0.0005
N_inputs=784
N_excitatory=int(argv[1])#400
tau_excitatory=0.1/tstep_logical
tau_inhibitory=0.01/tstep_logical
refrac_e=5e-3/tstep_logical
refrac_i=2e-3/tstep_logical
tau_e2e=1e-3/tstep_logical
tau_e2i=1e-3/tstep_logical
tau_i2e=2e-3/tstep_logical
tau_i2i=2e-3/tstep_logical
tau_pre_trace=0.02/tstep_logical
tau_post_trace=0.02/tstep_logical
w_ei=10.4
w_ie=17j
tau_theta=10000/tstep_logical
theta_delta=0.00005
theta_init=0.02
w_ee_colsum=78.0

input_steps=round(0.35/tstep_logical)
rest_steps=round(0.15/tstep_logical)
initial_freq=63.75*tstep_logical
additional_freq=32*tstep_logical
spikes_needed=5
batch_size=int(argv[2])#10000
epoch=int(argv[3])#{100:1, 400:3, 1600:7, 6400:15}[N_excitatory]

plastic_synapse_type=argv[4]
plastic_synapse_params=argv[5:]

flog=open(os.path.join('outputs_sweeps', '_'.join([s.replace('/','-') for s in argv[1:]])), 'w')

net=SpikingNet()
net.add_input('input')

excitatory=HomeostasisLayer(tau_theta, theta_delta, theta_init, N_excitatory, tau_excitatory, -52e-3-20e-3, -65e-3, -65e-3, -65e-3-40e-3, refrac_e, tau_e2e, tau_i2e, 0.1)
net.add_layer(excitatory, 'excitatory')

inhibitory=SecondOrderLayer(N_excitatory, tau_inhibitory, -40e-3, -45e-3, -60e-3, -60e-3-40e-3, refrac_i, tau_e2i, tau_i2i, 0.085)
net.add_layer(inhibitory, 'inhibitory')

winit=(np.random.random((N_excitatory, N_inputs))+0.01)*0.3
trace_pre=(1, np.exp(-1./tau_pre_trace))
trace_post=(1, np.exp(-1./tau_post_trace))
if plastic_synapse_type=='ideal':
    lr, =plastic_synapse_params
    lr=float(lr)
    syn_input_exc=PlasticSynapse(N_inputs, N_excitatory, winit, trace_pre, trace_post,
                            rule_params={'Ap':lr, 'An':0.01*lr}, Wlim=1, syn_type='exc', tracelim=1)
elif plastic_synapse_type=='Paiyu_chen_15':
    start_voltage, g1, nsubstep=plastic_synapse_params
    start_voltage=float(start_voltage)
    g1=float(g1) # was 1e-5
    nsubstep=int(nsubstep)
    # 0.075v = 10x dG/dt (same initial G)
    # 0.0716v/dec @ 1e-4S
    # 1.475v=10S/s @ 1e-5S
    # 1.336v=10S/s @ 1e-4S
    dv=0.073
    cell=cell_Paiyu_chen_15(cell_Paiyu_chen_15.w2t(winit, g1), g1, lambda te:np.maximum(start_voltage+np.log10(te)*dv, 0), lambda to:-np.maximum(start_voltage+(np.log10(to)-1)*dv, 0), 1e-9, 10e-9, nsubstep)
    syn_input_exc=SynapseCircuit(cell, N_inputs, N_excitatory, trace_pre, trace_post, syn_type='exc', tracelim=1)
elif plastic_synapse_type=='Vteam':
    lr, g1=plastic_synapse_params
    lr=float(lr)
    g1=float(g1)
    dRpot=-1/g1*lr
    dRdep=0.01/g1*lr
    cell=cell_Vteam(cell_Vteam.w2t(winit, g1), g1, lambda te:te*dRpot, lambda to:to*dRdep, 1e-9, 10e-9)
    syn_input_exc=SynapseCircuit(cell, N_inputs, N_excitatory, trace_pre, trace_post, syn_type='exc', tracelim=1)
elif plastic_synapse_type=='data-driven':
    fname, g1, v1_p, v1_n, dv_ep, dv_en = plastic_synapse_params
    g1=float(g1)
    v1_p=float(v1_p)
    v1_n=float(v1_n)
    dv_ep=float(dv_ep)# 1/95
    dv_en=float(dv_en)# 1/30
    # dep: e^30 per volt
    cell=cell_data_driven(winit, g1, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0), lambda to:-np.maximum(v1_n+np.log(to)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
    syn_input_exc=SynapseCircuit(cell, N_inputs, N_excitatory, trace_pre, trace_post, syn_type='exc', tracelim=1)
    syn_input_exc.W=winit
elif plastic_synapse_type=='none':
    syn_input_exc=BaseSynapse(N_inputs, N_excitatory, winit)
else:
    raise ValueError('unknown synapse type %s'%plastic_synapse_type)
net.add_synapse('excitatory', syn_input_exc, 'input')

syn_ei=OneToOneSynapse(N_excitatory, N_excitatory, np.array(w_ei))
net.add_synapse('inhibitory', syn_ei, 'excitatory')

syn_ie=LambdaSynapse(N_excitatory, N_excitatory, lambda x:w_ie*(x.sum(axis=-2 if x.ndim>1 else 0, keepdims=True)-x))
net.add_synapse('excitatory', syn_ie, 'inhibitory')

net.add_output("excitatory")

def normalize(W, target):
    W=np.clip(W, 0, 1)
    colsum=W.sum(axis=1)
    colfactors=target/colsum
    W*=colfactors[:,np.newaxis]
    return W

batch_rng=np.random.default_rng(seed=1)
def batchify(train_images, train_labels):
    totalsize=train_images.shape[0]
    sequence=np.arange(totalsize)
    batch_rng.shuffle(sequence)
    train_images_batched=train_images[sequence].reshape([-1, batch_size, train_images.shape[-1]])
    train_labels_batched=train_labels[sequence].reshape([-1, batch_size])
    return train_images_batched, train_labels_batched

mnist_file=np.load('mnist.npz')
(train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
train_images=train_images.reshape([train_images.shape[0], -1])/255
test_images=test_images.reshape([test_images.shape[0], -1])/255

previous_assignment=None
for batch, label_batch in itertools.chain(*([zip(*batchify(train_images, train_labels)) for i in range(epoch)]+[((test_images, test_labels),)])):
    assignment_matrix=np.zeros((10, N_excitatory))
    sample_correct=0
    stats=np.zeros(9)
    for sampleid, (sample, label) in enumerate(zip(batch, label_batch)):
        if sampleid>0 and sampleid%math.ceil(batch_size/10)==0:
            print('\t\t\t\t\t\tcorrect: %d/%d    '%(sample_correct, sampleid), end='\r')
        freq=initial_freq
        outputcnt=0
        while outputcnt<spikes_needed:
            sample_spike=Poisson(N_inputs, freq*sample)
            output_total=np.zeros(N_excitatory)
            outputcnt=0
            syn_input_exc.reset_stats()
            if previous_assignment is not None:
                prediction_vector=np.zeros(10)
            syn_input_exc.W=normalize(syn_input_exc.W, w_ee_colsum)
            for step in range(input_steps):
                outputs,=net(sample_spike())
                outputcnt+=outputs.sum()
                output_total+=outputs
                if previous_assignment is not None:
                    prediction_vector+=previous_assignment@outputs
            print('frequency %f, %f output spikes'%(freq, outputcnt), end='\r')
            #set_trace()
            for step in range(rest_steps):
                net(np.zeros(N_inputs))
            freq+=additional_freq
        assignment_matrix[label]+=output_total
        is_correct=0
        if previous_assignment is not None and prediction_vector.argmax()==label:
            sample_correct+=1
            is_correct=1
        stats+=[is_correct, syn_input_exc.power_forward, syn_input_exc.energy_update,
            syn_input_exc.count_input_spike, syn_input_exc.count_output_spike,
            syn_input_exc.sum_v_pre, syn_input_exc.sum_v2_pre, syn_input_exc.sum_v_post,
            syn_input_exc.sum_v2_post]
    # print('W: max %.4f, min %.4f, avg %.4f'%(syn_input_exc.W.max(), syn_input_exc.W.min(), syn_input_exc.W.mean()))
    # print('theta: max %.4f, min %.4f, avg %.4f'%(excitatory.theta.max(), excitatory.theta.min(), excitatory.theta.mean()))
    stats/=len(batch)
    print(*stats, file=flog, flush=True)
    if previous_assignment is not None:
        print('acc: %.2f%%'%(sample_correct/batch_size*100))
    previous_assignment=np.zeros((10, N_excitatory))
    label_frequency=np.eye(10)[label_batch].sum(axis=0)[:,np.newaxis]
    previous_assignment[(assignment_matrix/label_frequency).argmax(axis=0),range(N_excitatory)]=1
    previous_assignment/=previous_assignment.sum(axis=1, keepdims=True)
