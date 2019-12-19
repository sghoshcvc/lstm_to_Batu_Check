#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:32:14 2019

@author: pragna
"""
import random
import os
import pandas as pd
import collections
import numpy as np
import sys
from utils import utility
from costModel import matrixForm
from torch.autograd import Variable
from collections import defaultdict
import config
import torch
import torch.nn as nn
class training():
    def __init__(self, dirName, noAGV):
        self.dirName = dirName
        # self.dirName = rootDic
        self.base1 = "state"
        self.base2 = "P"
        self.base3 = 'sNext'
        self.base4 = 'PNext'
        self.suffix = '.txt'
        self.ownNo = noAGV
        self.allS = []
        self.allObs = []
        self.lastTask = 0
        self.loss_fn = nn.MSELoss()
        self.Model = matrixForm(5, dirName, self.ownNo)
        self.k = 0
       # self.stateDict = {}
        self.file1 = os.path.join(self.dirName, self.base1 + str(self.ownNo) + self.suffix)
        self.file2 = os.path.join(self.dirName, self.base2 + str(self.ownNo) + self.suffix)
        self.file3 = os.path.join(self.dirName, self.base3 + str(self.ownNo) + self.suffix)
        self.file4 = os.path.join(self.dirName, self.base4 + str(self.ownNo) + self.suffix)

        self.saveEstimate = np.empty([100,3])
        self.saveObs = np.empty([100, 3])
        #self.
        #self.batch_size = 5
                        #k, obs, currTask, utilObj
    def doTrain(self, itr, obs, currTask, prevTask, obsObj):

        self.k = itr
        #loss = nn.Crossentropy

        if (self.k>1):

           # maxKey = max(completedTask_lst, key=int)
            #lastTask = completedTask_lst[maxKey]
            PCurr, sCurr = obsObj.get_state(prevTask)
            outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(self.k) + ' ' + 'sCurr: shape and value: ' + str(sCurr.shape) + ' ' + str(sCurr) + '\n'
            fid1 = open(self.file1, 'a')
            fid1.write(outtxt1)
            fid1.close()

            outtxt2 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(self.k) + ' ' + 'h_next: shape and value: ' + str(PCurr.shape) + ' ' + str(PCurr) + '\n'
            fid2 = open(self.file2, 'a')
            fid2.write(outtxt2)
            fid2.close()
        elif (self.k == 1):

            obsObj.get_zero_state(k=0)
            print("Zero state: shape and value", obsObj.zero_state.shape, obsObj.zero_state)

            PCurr = obsObj.zero_P
            sCurr = obsObj.zero_state

            outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(self.k) + ' ' + 'sCurr: shape and value: ' + str(
                sCurr.shape) + ' ' + str(sCurr) + '\n'
            fid1 = open(self.file1, 'a')
            fid1.write(outtxt1)
            fid1.close()

            outtxt2 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(self.k) + ' ' + 'h_next: shape and value: ' + str(
                PCurr.shape) + ' ' + str(PCurr) + '\n'
            fid2 = open(self.file2, 'a')
            fid2.write(outtxt2)
            fid2.close()

            sCurr = torch.from_numpy(np.array(sCurr))
            sCurr = sCurr.to(dtype=torch.float32)
            sCurr = Variable(sCurr, requires_grad=True)
            #print(" sCurr er shape and value", sCurr.shape, sCurr)


        obs = torch.from_numpy(np.array(obs))
        obs = obs.to(dtype=torch.float32)
        obs = Variable(obs, requires_grad=True)
        #print(" obs: shape and value", obs.shape, obs)

        print("k: ", self.k)
        sNext, PNext = self.Model.findCost(self.k, obs, PCurr, sCurr)
        sNext = Variable(sNext, requires_grad=True)

        obsObj.store_state(PNext, sNext, currTask)
        _s_Next_ = sNext.data
        _s_Next_ = _s_Next_.numpy()

        #print("Shape of _s_Next_: ", _s_Next_.shape)



        #df = pd.DataFrame(_s_Next_)

        ## save to xlsx file

        #filepath = 'stateNext.xlsx'

        #df.to_excel(filepath, index=False)

        #fid3 = open(self.file3, 'a')
        #np.savetxt(fid3, _s_Next_, delimiter=',')
        #fid3.close()

        _P_Next_ = PNext.data
        _P_Next_ = _P_Next_.numpy()
        fid4 = open(self.file4, 'a')
        np.savetxt(fid4, _P_Next_, delimiter=',')
        fid4.close()

        self.allS.append(sNext)

        sSave = sNext.data.numpy()
        obsSave = obs.data.numpy()
        self.saveEstimate[(self.k-1)][:] = sSave
        self.saveObs[(self.k - 1)][:] = obsSave

        self.allObs.append(obs)

        self.lastTask = currTask
        #self.obsObj.storeObs(self.k, self.lastTask)
            
        return sSave

    #return P, s
    #for every k ==5, calculate loss
    # call forward
    
    def computeCost(self, rootDic, noAGV, k, u, utilObj, prevTask):
    #rootDic = sys.argv[1]
    #taskList = np.arange(100)
    #q = taskList
    #state = 'INITIALISE'

    #itr = 0

        currTask = u
    #trainObj = training(rootDic, noAGV)
            #obsObj = utility(rootDic)
        optim = torch.optim.Adam(self.Model.parameters())
            #lastTask = 0

        # if state == 'START':
        #     currTask = random.randint(0, 100)
        #     if (currTask in q):
        #         q= q[q != currTask]
        #         #q.remove()
        #         state = 'EXECUTE'
        #     elif (currTask not in q):
        #         state = 'START'
        # if state == 'EXECUTE':

            #itr = itr +1
        obs = utilObj.readObs(prevTask)# this iteration have to be calculated from planner
             # The last task is to be decided by the planner
            #allsTensor, allobsTensor = \
        estimatedCost = self.doTrain(k, obs, currTask, prevTask, utilObj) #itr, obs, currTask, prevTask, obsObj
        print('estimated Cost' + str(estimatedCost))

        #utilObj.storeObs(k, currTask)

        params = config.get_params()
        if (k >=5):
            if k % params["timestep"] == 0:
                print("Calculating Loss")
                allsTensor = torch.Tensor(self.k, 1, 3)
                allsTensor = torch.cat(self.allS, dim=1)
                              #out=)
                allobsTensor = torch.Tensor(self.k, 1, 3)
                allobsTensor = torch.cat(self.allObs, dim=1)
                L = self.loss_fn(allsTensor, allobsTensor)
                print('Loss =' + str(L))
                # sys.exit()

                L.backward()
                self.allObs.clear()
                self.allS.clear()
                
                optim.step()
                optim.zero_grad()

            #end if


        np.savetxt('estimate.xls', self.saveEstimate, delimiter=',')
        np.savetxt('observations.xls', self.saveObs, delimiter=',')

        #self.stateDict = collections.OrderedDict(sorted(self.stateDict.items()))

        return estimatedCost


    