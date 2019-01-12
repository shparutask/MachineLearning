import numpy as np
import dataset as data
import networks as nets

FILE_FOR_RESULTS = "../Dataset/results.txt"
f = open(FILE_FOR_RESULTS, 'w')

#Get data
train_data, train_labels, test_data, test_labels = data.get_dataset()

#Get networks
criterion, net_3layer, net_3layer_dropout, net_2layer, net_2layer_dropout = nets.get_Networks()

# Looping params
n_epoch = 20
batch_size = 60000

f.write("Values of loss function\n")
f.write("Layer\Epoch\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15\t16\t17\t18\t19\n")

loss_2layer = nets.loss_acc_net(net_2layer, batch_size, n_epoch, train_data, train_labels, criterion)
f.write("2-layer_loss: " + loss_2layer[0] + "\n")
f.write("2-layer_acc: " + loss_2layer[1] + "\n")

loss_3layer = nets.loss_acc_net(net_3layer, batch_size, n_epoch, train_data, train_labels, criterion)
f.write("3-layer_loss: " + loss_3layer[0] + "\n")
f.write("3-layer_acc: " + loss_3layer[1] + "\n")

loss_2layer_dropout = nets.loss_acc_net(net_2layer_dropout, batch_size, n_epoch, train_data, train_labels, criterion)
f.write("2-layer network with dropout_loss: " + loss_2layer_dropout[0] + "\n")
f.write("2-layer network with dropout_acc: " + loss_2layer_dropout[1] + "\n")

loss_3layer_dropout = nets.loss_acc_net(net_3layer_dropout, batch_size, n_epoch, train_data, train_labels, criterion)
f.write("3-layer network with dropout_loss: " + loss_3layer_dropout[0] + "\n")
f.write("3-layer network with dropout_acc: " + loss_3layer_dropout[1] + "\n")

f.close()