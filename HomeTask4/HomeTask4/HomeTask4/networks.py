import numpy as np

import Sequential as seq

import Linear as linear
import Soft_Max as softMax
import Batch_Normalization as batch
import Dropout as drop

import ReLU as r
import elu as elu
import leaky_ReLU as leaky
import Soft_Plus as softPlus

def get_Networks(data_size, predict_size):    
    ReLU_net = seq.Sequential()
    ReLU_net.add(linear.Linear(data_size, 100))
    ReLU_net.add(r.ReLU())
    ReLU_net.add(linear.Linear(100, 50))
    ReLU_net.add(r.ReLU())
    ReLU_net.add(linear.Linear(50, predict_size))
    ReLU_net.add(softMax.SoftMax())

    ELU_net = seq.Sequential()
    ELU_net.add(linear.Linear(data_size, predict_size))
    ELU_net.add(elu.ELU())
    ELU_net.add(softMax.SoftMax())

    LeakyReLU_net = seq.Sequential()
    LeakyReLU_net.add(linear.Linear(data_size, 400))
    LeakyReLU_net.add(leaky.LeakyReLU())
    LeakyReLU_net.add(linear.Linear(400, 250))
    LeakyReLU_net.add(leaky.LeakyReLU())
    LeakyReLU_net.add(linear.Linear(250, predict_size))
    LeakyReLU_net.add(leaky.LeakyReLU())
    LeakyReLU_net.add(softMax.SoftMax())

    SoftPlus_net = seq.Sequential()
    SoftPlus_net.add(linear.Linear(data_size, 30))
    SoftPlus_net.add(softPlus.SoftPlus())
    SoftPlus_net.add(linear.Linear(30, predict_size))
    SoftPlus_net.add(softMax.SoftMax())

    return ReLU_net, ELU_net, LeakyReLU_net, SoftPlus_net

def get_Networks_with_batch(data_size, predict_size):    
    ReLU_net = seq.Sequential()
    ReLU_net.add(linear.Linear(data_size, 100))
    ReLU_net.add(batch.BatchNormalization(0.3))
    ReLU_net.add(batch.ChannelwiseScaling(100))
    ReLU_net.add(r.ReLU())
    ReLU_net.add(linear.Linear(100, predict_size))
    ReLU_net.add(softMax.SoftMax())

    ELU_net = seq.Sequential()
    ELU_net.add(linear.Linear(data_size, predict_size))
    ELU_net.add(batch.BatchNormalization())
    ELU_net.add(batch.ChannelwiseScaling(predict_size))
    ELU_net.add(elu.ELU())
    ELU_net.add(softMax.SoftMax())

    return ReLU_net, ELU_net

def get_Networks_witn_Dropout(data_size, predict_size):    
    ReLU_net = seq.Sequential()
    ReLU_net.add(linear.Linear(data_size, 100))
    ReLU_net.add(batch.BatchNormalization(0.3))
    ReLU_net.add(batch.ChannelwiseScaling(100))
    ReLU_net.add(r.ReLU())
    ReLU_net.add(drop.Dropout())
    ReLU_net.add(linear.Linear(100, predict_size))
    ReLU_net.add(softMax.SoftMax())

    ELU_net = seq.Sequential()
    ELU_net.add(linear.Linear(data_size, predict_size))
    ELU_net.add(batch.BatchNormalization())
    ELU_net.add(batch.ChannelwiseScaling(predict_size))
    ELU_net.add(elu.ELU())
    ELU_net.add(drop.Dropout())
    ELU_net.add(linear.Linear(predict_size, predict_size))
    ELU_net.add(softMax.SoftMax())

    return ReLU_net, ELU_net