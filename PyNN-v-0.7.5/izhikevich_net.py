# -*- coding: utf-8 -*-

# Author(s): simonluca
# Created:   2014-04-02
# Version:   1.0

"""Izhikevich network.

This script re-implements the network model presented in [1]
using pyNN v-0.7.5 [2].

References

    [1] Izhikevich, Simple Model of Spiking Neurons,
        IEEE Transactions on Neural Networks (2003)

    [2] Davison et al., PyNN: A Common Interface for Neuronal Network
        Simulators, Frontiers in neuroinformatics (2009)

"""
import pyNN.nest as sim
import nest
import nest.raster_plot
import nest.voltage_trace
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


#----------- Build --------------
#--------------------------------
# Create Izhikevich cell type (based on native NEST model)
Izhikevich = sim.native_cell_type('izhikevich')

# Create excitatory neurons
r = np.random.rand(n_neurons_e)
c = -65.0 + 15.0 * r**2
d = 8.0 - 6.0 * r**2
population_e = sim.Population(n_neurons_e, Izhikevich,
                              cellparams={'a': 0.02,
                                          'b': 0.2})
population_e.tset('c', c)
population_e.tset('d', d)
population_e.initialize('V_m', -65.0)
population_e.initialize('U_m', 0.2*(-65.0))

# Create inhibitory neurons
r = np.random.rand(n_neurons_i)
a = 0.02 + 0.08 * r
b = 0.25 - 0.05 * r
population_i = sim.Population(n_neurons_i, Izhikevich,
                              cellparams={'c': -65.0,
                                          'd': 2.0})
population_i.tset('a', a)
population_i.tset('b', b)
population_i.initialize('V_m', -65.0)
population_i.initialize('U_m', b*(-65.0))

# Create assembly (entire population)
population = sim.Assembly(population_e, population_i)

# Create synaptic connections
syn_weight_e = max_syn_weight_e * np.random.rand(n_neurons_e, n_neurons)
syn_e = sim.Projection(population_e, population,
                       sim.AllToAllConnector(weights = syn_weight_e,
                                             delays = 1.0),
                       target='excitatory')
syn_weight_i = -max_syn_weight_i * np.random.rand(n_neurons_i, n_neurons)
syn_i = sim.Projection(population_i, population,
                       sim.AllToAllConnector(weights = syn_weight_i,
                                             delays = 1.0),
                       target='inhibitory')

# Create noise Gaussian generator
gauss_gen_e_params = {
    'mean': 0.0,  # pA
    'stdev': 0.005,  # pA
    'dt': 1.0,  # ms
    'rng': sim.NativeRNG()
    }
gauss_gen_i_params = {
    'mean': 0.0,  # pA
    'stdev': 0.002,  # pA
    'dt': 1.0,  # ms
    'rng': sim.NativeRNG()
    }
gauss_gen_e = sim.NoisyCurrentSource(**gauss_gen_e_params)
population_e.inject(gauss_gen_e)
gauss_gen_i = sim.NoisyCurrentSource(**gauss_gen_i_params)
population_i.inject(gauss_gen_i)

# Set spike times to be recorded
population_e.record()
population_i.record()

# Create and connect NEST(!) spike detector
spike_detec = nest.Create('spike_detector', 1)
nest.ConvergentConnect(list(population.all_cells), spike_detec)


#------------ Run ---------------
#--------------------------------
# Run simulation
sim.run(sim_time)


#----------- Plot ---------------
#--------------------------------
# Rasterplot
nest.raster_plot.from_device(spike_detec, "")
nest.raster_plot.show()
plt.axis([0, sim_time, population[0]-1, population[-1:][0]+1])
plt.title('Raster')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')


#------ Simulation End ----------
#--------------------------------
# Clean up simulation framework
sim.end()
