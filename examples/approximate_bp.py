import numpy as np
import gzip
import pickle
from sys import argv, path
import os

path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))

from spikelearn import SpikingNet, IntFireLayer, PlasticSynapse, LambdaSynapse, cell_Paiyu_chen_15, SynapseCircuitBP, cell_Vteam, cell_data_driven_bp, cell_data_driven_bp_T
from spikelearn.generators import Poisson
from spikelearn.trace import ManualTrace

T=50
update_interval=7

# C1=64
# C2=32
# suffix='init_1'
# features_X=np.load(open(f'stdp_conv_features_{C1}-{C2}_{suffix}.npz','rb'))['features_X']
# mnist_file=np.load('mnist.npz')
# (train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
# Tconv=int(argv[4]) # 40

# F=pickle.load(gzip.open('../Spiking-CNN/EndToEnd_STDP_Spiking_CNN/RL8_Maps_1632.pickle.gz','rb'))
# print('!!!!!!!!!!     Using original feature maps     !!!!!!!!!!')
# X=np.array([x[1] for x in F['Maps']])
# features_X=X.reshape((X.shape[0], -1))
# train_labels=F['Labels'].astype(np.int8)
# Tconv=20

mnist_file=np.load('mnist.npz')
(train_images, train_labels), (test_images, test_labels)=(mnist_file['x_train'], mnist_file['y_train']), (mnist_file['x_test'], mnist_file['y_test']) # tf.keras.datasets.mnist.load_data()
features_X=train_images.reshape((train_images.shape[0], -1))
# print('!!!!!!!!!!     Using raw mnist     !!!!!!!!!!')
Tconv=255

N_samples, N_input = features_X.shape
print(f'max frequency {features_X.max()}/{Tconv}')

plastic_synapse_type=argv[1]
plastic_synapse_params=argv[2:]

W1=np.clip(np.random.randn(1500, N_input)+5, 0, 10)*0.01
W2=np.clip(np.random.randn(10, 1500)+5, 0, 10)*0.01
trace_L1_pre=ManualTrace(N_input, 0, np.int32)
trace_L1_post=ManualTrace(1500, 0, np.int32)
trace_L2_pre=ManualTrace(1500, 0, np.int32)
trace_L2_post=ManualTrace(10, 0, np.int32)
if plastic_synapse_type=='ideal':
    syn_L1=PlasticSynapse(N_input, 1500, W1, trace_L1_pre, trace_L1_post, Wlim=0.1, syn_type='exc', rule_params={'Ap':0, 'An':-0.0002})
    syn_L2=PlasticSynapse(1500, 10, W2, trace_L2_pre, trace_L2_post, Wlim=0.1, syn_type='exc', rule_params={'Ap':0, 'An':-0.0002})
elif plastic_synapse_type=='Paiyu_chen_15':
    start_voltage, g1, nsubstep=plastic_synapse_params
    start_voltage=float(start_voltage)
    g1=float(g1)
    nsubstep=int(nsubstep)
    # 0.075v = 10x dG/dt (same initial G)
    # 0.0716v/dec @ 1e-4S
    # 1.475v=10S/s @ 1e-5S
    # 1.336v=10S/s @ 1e-4S
    dv=0.073
    cell1=cell_Paiyu_chen_15(cell_Paiyu_chen_15.w2t(W1, g1), g1, lambda te:np.maximum(start_voltage+np.log10(te*2)*dv, 0), lambda to:-np.maximum(start_voltage+np.log10(to*2)*dv, 0), 1e-9, np.nan, nsubstep)
    syn_L1=SynapseCircuitBP(cell1, N_input, 1500, trace_L1_pre, trace_L1_post, syn_type='exc', tracelim=None)
    cell2=cell_Paiyu_chen_15(cell_Paiyu_chen_15.w2t(W2, g1), g1, lambda te:np.maximum(start_voltage+np.log10(te*2)*dv, 0), lambda to:-np.maximum(start_voltage+np.log10(to*2)*dv, 0), 1e-9, np.nan, nsubstep)
    syn_L2=SynapseCircuitBP(cell2, 1500, 10, trace_L2_pre, trace_L2_post, syn_type='exc', tracelim=None)
elif plastic_synapse_type=='Vteam':
    lr, g1=plastic_synapse_params
    lr=float(lr)
    g1=float(g1)
    dRpot=-1/g1*lr
    dRdep=1/g1*lr
    cell1=cell_Vteam(cell_Vteam.w2t(W1, g1), g1, lambda te:te*dRpot, lambda to:to*dRdep, 1e-9, 10e-9)
    syn_L1=SynapseCircuitBP(cell1, N_input, 1500, trace_L1_pre, trace_L1_post, syn_type='exc', tracelim=None)
    cell2=cell_Vteam(cell_Vteam.w2t(W2, g1), g1, lambda te:te*dRpot, lambda to:to*dRdep, 1e-9, 10e-9)
    syn_L2=SynapseCircuitBP(cell2, 1500, 10, trace_L2_pre, trace_L2_post, syn_type='exc', tracelim=None)
elif plastic_synapse_type=='1T1R':
    g1, v1_p, v1_n = plastic_synapse_params
    g1=float(g1)
    v1_p=float(v1_p)
    v1_n=float(v1_n)
    dv_ep=1/95
    dv_en=1/30
    fname='model_data/1t1r_Paiyu_15.npz'
    # dep: e^30 per volt
    cell1=cell_data_driven_bp(W1, g1, lambda to:np.maximum(v1_n+np.log(to*20)*dv_en, 0), lambda to:-np.maximum(v1_n+np.log(to*20)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
    syn_L1=SynapseCircuitBP(cell1, N_input, 1500, trace_L1_pre, trace_L1_post, syn_type='exc', tracelim=1)
    syn_L1.W=W1
    cell2=cell_data_driven_bp(W2, g1, lambda to:np.maximum(v1_n+np.log(to*20)*dv_en, 0), lambda to:-np.maximum(v1_n+np.log(to*20)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
    syn_L2=SynapseCircuitBP(cell2, 1500, 10, trace_L2_pre, trace_L2_post, syn_type='exc', tracelim=1)
    syn_L2.W=W2
elif plastic_synapse_type=='1T1R_T':
    g1, v1_p, v1_n = plastic_synapse_params
    g1=float(g1)
    v1_p=float(v1_p)
    v1_n=float(v1_n)
    dv_ep=1/95
    dv_en=1/30
    fname='model_data/1t1r_Paiyu_15.npz'
    # dep: e^30 per volt
    cell1=cell_data_driven_bp_T(W1, g1, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0), lambda to:-np.maximum(v1_n+np.log(to*20)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
    syn_L1=SynapseCircuitBP(cell1, N_input, 1500, trace_L1_pre, trace_L1_post, syn_type='exc', tracelim=1)
    syn_L1.W=W1
    cell2=cell_data_driven_bp_T(W2, g1, lambda te:np.maximum(v1_p+np.log(te)*dv_ep, 0), lambda to:-np.maximum(v1_n+np.log(to*20)*dv_en, 0), 1e-9, 10e-9, fname, Vread=0.7)
    syn_L2=SynapseCircuitBP(cell2, 1500, 10, trace_L2_pre, trace_L2_post, syn_type='exc', tracelim=1)
    syn_L2.W=W2
else:
    raise ValueError('unknown synapse type %s'%plastic_synapse_type)

net=SpikingNet()
net.add_input('input')
#seeds# np.random.seed(19)
syn_L1_bal=LambdaSynapse(N_input, 1500, lambda x:x.sum()*(-0.01*5)*np.ones(1500))
neuron_L1=IntFireLayer(1500, 0.1)
net.add_layer(neuron_L1, 'L1')
net.add_synapse('L1', syn_L1, 'input')
net.add_synapse('L1', syn_L1_bal, 'input')
syn_L2_bal=LambdaSynapse(1500, 10, lambda x:x.sum()*(-0.01*5)*np.ones(10))
neuron_L2=IntFireLayer(10, 1.0)
net.add_layer(neuron_L2, 'L2')
net.add_synapse('L2', syn_L2, 'L1')
net.add_synapse('L2', syn_L2_bal, 'L1')
net.add_output('L1')
net.add_output('L2')

flog=open(os.path.join('outputs_sweeps_bp', '_'.join([s.replace('/','-') for s in argv[1:]])), 'w')

TestA, TestB = 33000, 34000
Ntrain=33000
BatchTrain=1000

stats1=np.zeros(8)
stats2=np.zeros(8)
for iteration in range(3):
    #seeds# np.random.seed(iteration)
    for i in range(Ntrain):
        if i%BatchTrain==0 and (i>0 or iteration>0):
            pred=np.zeros(BatchTrain)
            for i2 in range(TestA,TestB):
                sample_spike=Poisson(N_input, features_X[i2], 1/Tconv)
                out_spikes_total=np.zeros(10)
                neuron_L1.reset()
                neuron_L2.reset()
                for t in range(T):
                    net.forward(sample_spike())
                    hidden_spikes,out_spikes=net.get_output()
                    out_spikes_total+=out_spikes
                pred[i2-TestA]=np.argmax(out_spikes_total)
            # print('acc:',np.mean(pred==train_labels[TestA:TestB]))
            acc=np.mean(pred==train_labels[TestA:TestB])
            stats1/=BatchTrain
            stats2/=BatchTrain
            print(acc, *stats1, *stats2, file=flog, flush=True)
            stats1[:]=0
            stats2[:]=0
        sample_spike=Poisson(N_input, features_X[i], 1/Tconv)
        neuron_L1.reset()
        neuron_L2.reset()
        syn_L1.tro.reset()
        syn_L2.tro.reset()
        syn_L1.tre.reset()
        syn_L2.tre.reset()
        syn_L1.reset_stats()
        syn_L2.reset_stats()
        expected=np.eye(10)[train_labels[i]]
        sample_spike()
        for t in range(1,T):
            input_spikes=sample_spike()
            net.forward(input_spikes)
            syn_L2.tro.t+=neuron_L2.s
            syn_L1.tro.t+=neuron_L1.s
            syn_L2.tre.t+=neuron_L1.s
            syn_L1.tre.t+=input_spikes
            if (t)%update_interval==0:
                input_spikes=np.zeros(N_input,dtype=np.int32)
                net.forward(input_spikes)
                syn_L2.tro.t+=neuron_L2.s
                syn_L1.tro.t+=neuron_L1.s
                syn_L2.tre.t+=neuron_L1.s
                syn_L1.tre.t+=input_spikes

                syn_L2.tro.t=expected-np.clip(syn_L2.tro.t, 0, 1)
                syn_L2.xe=syn_L2.tre.t#np.clip(syn_L2.tre.t, 0, 1)

                syn_L1.tro.t=(syn_L2.tro.t@(syn_L2.W-0.01*5))*np.where(syn_L2.tre.t>0,1,0)
                syn_L1.xe=syn_L1.tre.t#np.clip(syn_L1.tre.t, 0, 1)

                syn_L2.update(np.zeros(syn_L2.No))
                syn_L1.update(np.zeros(syn_L1.No))

                #print(syn_L2.xe.mean(), syn_L2.tro.t.mean(), syn_L1.xe.mean(), syn_L1.tro.t.mean())
                syn_L1.tro.reset()
                syn_L2.tro.reset()
                syn_L1.tre.reset()
                syn_L2.tre.reset()
        
        stats1+=[syn_L1.power_forward, syn_L1.energy_update,
            syn_L1.count_input_spike, syn_L1.count_output_spike,
            syn_L1.sum_v_pre, syn_L1.sum_v2_pre, syn_L1.sum_v_post,
            syn_L1.sum_v2_post]
        stats2+=[syn_L2.power_forward, syn_L2.energy_update,
            syn_L2.count_input_spike, syn_L2.count_output_spike,
            syn_L2.sum_v_pre, syn_L2.sum_v2_pre, syn_L2.sum_v_post,
            syn_L2.sum_v2_post]
