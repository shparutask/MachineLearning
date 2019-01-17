import numpy as np
import matplotlib.pyplot
from IPython import display
import matplotlib.pyplot as plt

import dataset as data
import networks as nets
import Helper as help

import MSECriterion as mse
import ClassNLLCriterionUnstable as nlls

X, Y = data.get_dataset()
ReLU_net, ELU_net, LeakyReLU_net, SoftPlus_net = nets.get_Networks(784, 10)

n_epoch = 20
batch_size = 1000

help.run_network(X, Y, ReLU_net, mse.MSECriterion(), n_epoch, batch_size)
help.run_network(X, Y, ELU_net, mse.MSECriterion(), n_epoch, batch_size)
help.run_network(X, Y, LeakyReLU_net, mse.MSECriterion(), n_epoch, batch_size)
help.run_network(X, Y, SoftPlus_net, mse.MSECriterion(), n_epoch, batch_size)

ReLU_net, ELU_net = nets.get_Networks_with_batch(784, 10)

help.run_network(X, Y, ReLU_net, mse.MSECriterion(), n_epoch, batch_size)
help.run_network(X, Y, ELU_net, mse.MSECriterion(), n_epoch, batch_size)

ReLU_net, ELU_net = nets.get_Networks_witn_Dropout(784, 10)

help.run_network(X, Y, ReLU_net, mse.MSECriterion(), n_epoch, batch_size)
help.run_network(X, Y, ELU_net, mse.MSECriterion(), n_epoch, batch_size)