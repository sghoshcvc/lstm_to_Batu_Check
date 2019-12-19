#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:10:25 2017

@author: tecnicmise
"""
#import numpy as np
from utils import utility
import os
import pandas as pd
import collections
#from bModel import bModel
from planning import K_nearestNeighbor
import time
import config
class AGV(K_nearestNeighbor):
    "Creating the topo map from the given grid map"
    
    def __init__(self, rootDic, noAGV):
        self.itr = 0
        self.ownNo = noAGV
        self.completedTask_lst = {}
        self.dirName = rootDic
        self.base1 = "tasks"
        self.suffix = '.txt'
        self.ownNo = noAGV
        self.f1 = os.path.join(self.dirName, self.base1 + str(self.ownNo) + self.suffix)

        K_nearestNeighbor.__init__(self, rootDic, noAGV)
    #end defn
    def scheduling(self):
        params = config.get_params()
        state = 'INITIALISE'
        while self.itr<=params["repNo"]:
            if state == 'INITIALISE':
                utilObj = utility(self.dirName, self.ownNo)
                if (self.ownNo == 1):
                    lastTask = 2
                    utilObj.storeCurrLoc(self.ownNo, lastTask)
                    #utilObj.storeObs(self.itr , lastTask)
                    utilObj.storeObs(k=1, currTask=lastTask, estCost=0)
                elif (self.ownNo == 2):
                    lastTask = 8
                    utilObj.storeCurrLoc(self.ownNo, lastTask)
                    utilObj.storeObs(k=1, currTask=lastTask, estCost=0)
                    #utilObj.storeCurrLoc(self.ownNo, 8)
                elif (self.ownNo == 3):
                    lastTask = 12
                    utilObj.storeCurrLoc(self.ownNo, lastTask)
                    utilObj.storeObs(k=1, currTask=lastTask, estCost=0)
                    #utilObj.storeCurrLoc(self.ownNo, 12)
                elif (self.ownNo == 4):
                    lastTask = 7
                    utilObj.storeCurrLoc(self.ownNo, lastTask)
                    utilObj.storeObs(k=1, currTask=lastTask, estCost=0)
                    #utilObj.storeCurrLoc(self.ownNo, 7)
                #end if

                state = 'START'
            if state == 'START':
                self.itr += 1
                currLoc = utilObj.getCurrLoc(self.ownNo)
                print("Itr: ", self.itr)
                print("currLoc at AGV level: ", currLoc)
                stateDict, lenTaskSeq = self.planning(currLoc, utilObj, params)
                stateDict = collections.OrderedDict(sorted(stateDict.items()))
                #print(stateDict)
                df = pd.DataFrame({key: pd.Series(value) for key, value in stateDict.items()})
                df.to_csv('estState.xlsx', encoding='utf-8', index=False)
                #
                #df = pd.DataFrame(data=stateDict, index=[0])

                #df = (df.T)

                #print(df)

                #df.to_excel()
                #print("Tasksequence of AGV", self.ownNo, self.taskSequence)
                self.completedTask_lst[self.itr] = self.taskSequence
                state = "COMPLETE"
            if state == 'COMPLETE':
                lastTask = self.endTask

                utilObj.storeCurrLoc(self.ownNo, lastTask)
                #utilObj.storeObs(self.lenT, self.endTask)
                outtxt1 = 'AGV: '+ str(self.ownNo) + ' ' + 'ITR: '+ str(self.itr) + ' ' + 'TASKS: ' + str(self.completedTask_lst) + '\n'
                self.fid1 = open(self.f1, 'a')
                self.fid1.write(outtxt1)
                self.fid1.close()
                state = 'START'
            #end if
        #end of while
    #end of scehduling