import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

import matplotlib.pyplot as plt
import numpy as np
import random

from dataset import DatasetProcessing
from models import SIAMESE

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for JHU CS661 Computer Vision HW3.")

parser.add_argument("--load", 
                    help="Load saved network weights. (default = best_weights)")
parser.add_argument("--save", 
                    help="Save network weights. (default = cnn_weight)")  
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")  
parser.add_argument("--learning_rate", "-lr", default=1e-2, type=float,
                    help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")               
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need


def main(options):
    # Path configuration
    TRAINING_PATH = '/home/ye/Works/googleCloud-SDK/train.txt'
    TESTING_PATH = '/home/ye/Works/googleCloud-SDK/test.txt'
    IMG_PATH = '/home/ye/Works/googleCloud-SDK/lfw'

    transformations = transforms.Compose([transforms.Scale((128,128)),
                                    transforms.ToTensor()
                                    ])
    
    dset_train = DatasetProcessing(IMG_PATH, TRAINING_PATH, transformations)

    dset_test = DatasetProcessing(IMG_PATH, TESTING_PATH, transformations)

    train_loader = DataLoader(dset_train,
                              batch_size = options.batch_size,
                              shuffle = True,
                             )

    test_loader = DataLoader(dset_test,
                             batch_size = options.batch_size,
                             shuffle = False,
                             )

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        cuda.set_device(options.gpuid[0])
    
    # Initial the model
    cnn_model = SIAMESE()
    if use_cuda > 0:
        cnn_model.cuda()
    else:
        cnn_model.cpu()

    # Binary cross-entropy loss
    criterion = torch.nn.BCELoss()
    optimizer = eval("torch.optim." + options.optimizer)(cnn_model.parameters())

    # main training loop
    last_dev_avg_loss = float("inf")
    for epoch_i in range(options.epochs):
        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss = 0.0
        correct_prediction = 0.0
        for it, train_data in enumerate(train_loader, 0):
            img0, img1, labels = train_data
            if use_cuda:
                img0, img1 , labels = Variable(img0).cuda(), Variable(img1).cuda() , Variable(labels).cuda()
            else:
                img0, img1 , labels = Variable(img0), Variable(img1), Variable(labels)

            train_output = cnn_model(img0,img1)
            loss = criterion(train_output, labels)
            train_loss += loss.data[0]
            predict = torch.round(train_output)
            correct_prediction += (predict.view(-1) == labels.view(-1)).sum().float()
            logging.debug("loss at batch {0}: {1}".format(it, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
        training_accuracy = (correct_prediction / len(dset_train)).data.cpu().numpy()[0]
        logging.info("Average training loss value per instance is {0} at the end of epoch {1}".format(train_avg_loss, epoch_i))
        logging.info("Training accuracy is {0} at the end of epoch {1}".format(training_accuracy, epoch_i))

        # validation -- this is a crude esitmation because there might be some paddings at the end
        dev_loss = 0.0
        correct_prediction = 0.0
        for it, test_data in enumerate(test_loader, 0):
            img0, img1, labels = test_data

            if use_cuda:
                img0, img1 , labels = Variable(img0, volatile=True).cuda(), Variable(img1, volatile=True).cuda() , Variable(labels, volatile=True).cuda()
            else:
                img0, img1 , labels = Variable(img0, volatile=True), Variable(img1, volatile=True), Variable(labels, volatile=True)

            test_output = cnn_model(img0,img1)
            loss = criterion(test_output, labels)
            dev_loss += loss.data[0]
            predict = torch.round(test_output)
            correct_prediction += (predict.view(-1) == labels.view(-1)).sum().float()

        dev_avg_loss = dev_loss / (len(dset_test) / options.batch_size)
        testing_accuracy = (correct_prediction / len(dset_test)).data.cpu().numpy()[0]
        logging.info("Average validation loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss, epoch_i))
        logging.info("Validation accuracy is {0} at the end of epoch {1}".format(testing_accuracy, epoch_i))

        #if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
        #    logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
        #    break
        #torch.save(cnn_model.state_dict(), open(options.save + ".nll_{0:.2f}.epoch_{1}".format(dev_avg_loss.data[0], epoch_i), 'wb'))
        last_dev_avg_loss = dev_avg_loss


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)