# -*- coding: utf-8 -*-

# Author(s): @simonluca
# Created:   2014-04-23
# Version:   1.0

"""Izhikevich network.

This script re-implements the network model presented in [1]
using pyNN v-0.8beta1 [2].

References

    [1] Izhikevich, Simple Model of Spiking Neurons,
        IEEE Transactions on Neural Networks (2003)

    [2] Davison et al., PyNN: A Common Interface for Neuronal Network
        Simulators, Frontiers in neuroinformatics (2009)

"""
import pyNN.nest as sim
import numpy as np
import matplotlib.pyplot as plt

#--------- Parameters -----------
#--------------------------------
# Simulation parameters
sim_time = 1000.0  # ms
resolution = 0.1  # ms
n_threads = 2

# Initialize simulation framework
sim.setup(timestep=resolution, min_delay=1.0, threads=n_threads)

# Set population dimensions
n_neurons_e = 800;
n_neurons_i = 200;
n_neurons = n_neurons_e + n_neurons_i;

# Set synaptic weights
max_syn_weight_e = 0.0005  # max magnitude of excitatory synaptic weights
max_syn_weight_i = 0.001   # max magnitude of inhibitory synaptic weights

# Set excitatory neurons parameters
r = np.random.rand(n_neurons_e)
a = 0.02
b = 0.2
c = -65.0 + 15.0 * r**2
d = 8.0 - 6.0 * r**2
v = -65.0
parameters_e = {'a': a,
                'b': b,
                'c': c,
                'd': d
                }
initialize_e = {'v': v,
                'u': b * v
                }

# Set inhibitory neurons parameters
r = np.random.rand(n_neurons_i)
a = 0.02 + 0.08 * r
b = 0.25 - 0.05 * r
c = -65.0
d = 2.0
v = -65.0
parameters_i = {'a': a,
                'b': b,
                'c': c,
                'd': d
                }
initialize_i = {'v': v,
                'u': b * v
                }

# Gaussian noise parameters
gauss_gen_e_params = {
    'mean': 0.0,  # nA
    'stdev': 0.005,  # nA
    'dt': 1.0,  # ms
    }
gauss_gen_i_params = {
    'mean': 0.0,  # nA
    'stdev': 0.002,  # nA
    'dt': 1.0,  # ms
    }


#----------- Build --------------
#--------------------------------
# Create neurons
population_e = sim.Population(n_neurons_e,
                              sim.Izhikevich(**parameters_e),
                              initial_values=initialize_e,
                              label="Ex")
population_i = sim.Population(n_neurons_i,
                              sim.Izhikevich(**parameters_i),
                              initial_values=initialize_i,
                              label="In")

# Assembly entire population
population = sim.Assembly(population_e, population_i)

# Create synapses
syn_weight_e = max_syn_weight_e * np.random.rand(n_neurons_e, n_neurons)
synapses_e = sim.Projection(population_e, population,
                            sim.AllToAllConnector(allow_self_connections=False),
                            sim.StaticSynapse(weight=syn_weight_e,
                                              delay=1.0),
                            receptor_type='excitatory')
syn_weight_i = -max_syn_weight_i * np.random.rand(n_neurons_i, n_neurons)
synapses_i = sim.Projection(population_i, population,
                            sim.AllToAllConnector(allow_self_connections=False),
                            sim.StaticSynapse(weight=syn_weight_i,
                                              delay=1.0),
                            receptor_type='inhibitory')

# Create noise Gaussian generator
gauss_gen_e = sim.NoisyCurrentSource(**gauss_gen_e_params)
population_e.inject(gauss_gen_e)
gauss_gen_i = sim.NoisyCurrentSource(**gauss_gen_i_params)
population_i.inject(gauss_gen_i)

# Set spike times to be recorded
population.record('spikes')


#------------ Run ---------------
#--------------------------------
# Run simulation
sim.run(sim_time)


#----------- Plot ---------------
#--------------------------------
# Get data
spikes = population.get_data()
voltage_trace_e = population_e.get_data()
voltage_trace_i = population_i.get_data()
segment = spikes.segments[0]
plt.figure(1)
# Raster
for n in np.arange(segment.size()['spiketrains']):
    if segment.spiketrains[n].annotations['source_population'] == 'Ex':
        color = 'r'
    else:
        color = 'b'
    y = np.ones_like(segment.spiketrains[n]) * (n+1)
    plt.plot(segment.spiketrains[n], y, color+'.')
plt.ylabel('Neuron #')
plt.yticks((0, n_neurons_e, n_neurons))
plt.xlabel('Time [ms]')
plt.xticks((0, sim_time))
plt.title(('Ex MFR = '+str("%.1f" % population_e.meanSpikeCount())+ \
           '   In MFR = '+str("%.1f" % population_i.meanSpikeCount())+ \
           '  [spikes/s]'))
plt.show()

#------ Simulation End ----------
#--------------------------------
# Clean up simulation framework
sim.end()
