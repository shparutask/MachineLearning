import numpy as np
import matplotlib.pyplot
from IPython import display
import matplotlib.pyplot as plt

import dataset as data
import networks as nets
import Helper as help

import MSECriterion as mse
import ClassNLLCriterionUnstable as nlls

train_data, train_labels, test_data, test_labels = data.get_dataset()
#ReLU_net, ELU_net, LeakyReLU_net, SoftPlus_net = nets.get_Networks(784, 10)

n_epoch = 20
batch_size = 1000

#help.train_network(train_data, train_labels, ReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.train_network(train_data, train_labels, ELU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.train_network(train_data, train_labels, LeakyReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.train_network(train_data, train_labels, SoftPlus_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)

ReLU_net, ELU_net = nets.get_Networks_with_batch(784, 10)

#help.train_network(train_data, train_labels, ReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.train_network(train_data, train_labels, ELU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)

ReLU_net, ELU_net = nets.get_Networks_witn_Dropout(784, 10)

#help.train_network(train_data, train_labels, ReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
help.train_network(train_data, train_labels, ELU_net,nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)


#help.test_network(test_data, test_labels, ReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.test_network(test_data, test_labels, ELU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.test_network(test_data, test_labels, LeakyReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.test_network(test_data, test_labels, SoftPlus_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)

#ReLU_net, ELU_net = nets.get_Networks_with_batch(784, 10)

#help.test_network(test_data, test_labels, ReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
#help.test_network(test_data, test_labels, ELU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)

#ReLU_net, ELU_net = nets.get_Networks_witn_Dropout(784, 10)
#help.test_network(test_data, test_labels, ReLU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)
help.test_network(test_data, test_labels, ELU_net, nlls.ClassNLLCriterionUnstable(), n_epoch, batch_size)