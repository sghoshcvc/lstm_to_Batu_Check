#from bilinear import bilinear_model
from Kf import kalman
import torch
import torch.nn as nn
from torch.autograd import Variable
from statistics import mean
from collections import defaultdict#, OrderedDict
import os
import math
import numpy as np
import pickle
#from obsMatrix import observationMean
#from readObs import Obs
#from addNewTT import newTT
#import time
#from addNewTT import addNewTT, fetchOtherObs
import time
import torch.nn.functional as Fn


class matrixForm(nn.Module):
    def __init__(self, regN, dirName, noAGV):

        super(matrixForm, self).__init__()
        self.input_dim = 3
        self.hidden_dim = 6
        self.num_layers = 1
        self.outputDim = 3
        self.ownNo = noAGV
        #self.noutputs = 6
        self.h_next = 0
        self.c_next = 0
        self.kfObj = kalman(regN)
        self.lstm_f = nn.LSTMCell(self.input_dim, self.hidden_dim, self.num_layers)
        self.lstm_q = nn.LSTMCell(self.input_dim, self.hidden_dim, self.num_layers)
        self.lstm_r = nn.LSTMCell(self.input_dim, self.hidden_dim, self.num_layers)

        self.fc_out_f = nn.Linear(self.hidden_dim, self.outputDim)
        self.fc_out_q = nn.Linear(self.hidden_dim, self.outputDim)
        self.fc_out_r = nn.Linear(self.hidden_dim, self.outputDim)

        # nn.init.zeros_(self.fc_out_f.weight)
        # nn.init.zeros_(self.fc_out_q.weight)
        # nn.init.zeros_(self.fc_out_r.weight)

        self.dirName = dirName
        self.base1 = 'h_next'
        self.base2 = 'c_next'
        self.base3 = 's_apriori'
        self.base4 = 'Est_Q'
        self.base5 = 'Est_R'
        self.base6 = 'P_'
        self.base7 = 'F'

        #self.no = str(ownNo)
        self.suffix = '.txt'
        self.file1 = os.path.join(self.dirName, self.base1 + str(self.ownNo) + self.suffix)
        self.file2 = os.path.join(self.dirName, self.base2 + str(self.ownNo) + self.suffix)
        self.file3 = os.path.join(self.dirName, self.base3 + str(self.ownNo) + self.suffix)
        self.file4 = os.path.join(self.dirName, self.base4 + str(self.ownNo) + self.suffix)
        self.file5 = os.path.join(self.dirName, self.base5 + str(self.ownNo) + self.suffix)
        self.file6 = os.path.join(self.dirName, self.base6 + str(self.ownNo) + self.suffix)
        self.file7 = os.path.join(self.dirName, self.base7 + str(self.ownNo) + self.suffix)

    def get_jacobian(self, net, fc, x):
        #print("Input inside jacobian: x shape and value", x.shape, x)
        y, z = net(x)

        y = fc(y)
        gradients = x
        y.backward(gradients)#(torch.eye(self.outputDim))
        return x.grad.data
                 #(self,regNo,inPCurr,seqNeigh,currNeighbor,currNode,tRegd,currentSource,bModObj)
    def findCost(self, k, obs, PCurr, sCurr):
        #########################################################################
        ### finding cost for each task using LSTM        ###
        ### 3 LSTMs are there, LSTM_f is for state_apriori, LSTM_q ########
        ### and LSTM_r are for estimating Q and R
        #### F is the Jacoian of f (LSTM function) w.r.t s_apriori ######
        #########################################################################
            #print("k inside findCost", k)
            #if (k ==1):
                #x = torch.from_numpy(np.array(sCurr))
                #inputX = x.to(dtype=torch.float32)
                #final_input = Variable(inputX, requires_grad=True)
                #print(" x er shape and value", final_input.shape, final_input)
            #elif k >1:
                #print(" sCurr er shape and value", sCurr.shape, sCurr)
                #inputX = sCurr.to(dtype=torch.float32)


            #final_input = final_input.unsqueeze(0)
            
            #z = torch.from_numpy(np.array(obsCost))
            #zObs = z.to(dtype=torch.float32)
            #final_z= Variable(zObs, requires_grad=True)
            #print(" z er shape nad value", final_z.shape, final_z)
            final_input = sCurr
            final_z = obs
            if k == 1:

                h_0 = np.zeros([1, 6])
                #y = 1 / math.sqrt(1)
                #h_0[0][0] = y
                #h_0[0][1] = 1 - y
                #h_0[0][2] = 0.8 - y
                #h_0[0][3] = 0.7 - y
                #h_0[0][4] = 0.6 - y
                #h_0[0][5] = 0.5 - y
                h_0 = torch.from_numpy(h_0)
                h_0 = h_0.to(dtype=torch.float32)
                h_0 = Variable(h_0, requires_grad=True)

                c_0 = np.zeros([1, 6])
                #c_0[0][0] = -y
                #c_0[0][1] = 1 - y
                #c_0[0][2] = 0.9 - y
                #c_0[0][3] = 0.5 - y
                #c_0[0][4] = 0.6 - y
                #c_0[0][5] = 0.5 - y
                c_0 = torch.from_numpy(c_0)
                c_0 = c_0.to(dtype=torch.float32)
                c_0 = Variable(c_0, requires_grad=True)

                self.h_next, self.c_next = self.lstm_f(final_input, (h_0,c_0))
                #print("h_next: shape ", self.h_next)
                #print("c_next: shape and value", self.c_next)
                outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'h_next: shape and value: ' + str(self.h_next.shape) + ' ' + str(self.h_next) + '\n'
                fid1 = open(self.file1, 'a')
                fid1.write(outtxt1)
                fid1.close()

                outtxt2 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'c_next: shape and value: ' + str(self.c_next.shape) + ' ' + str(self.c_next) + '\n'
                fid2 = open(self.file2, 'a')
                fid2.write(outtxt2)
                fid2.close()

                s_apriori = Fn.relu(self.fc_out_f(Fn.relu(self.h_next)))
                outtxt3 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' '+ 's_apriori: shape and value' + str(s_apriori.shape) + ' ' + str(s_apriori) + '\n'
                fid3 = open(self.file3, 'a')
                fid3.write(outtxt3)
                fid3.close()
                #print("S_apriori: shape and value", s_apriori.shape, s_apriori)

                self.h_q_next, self.c_q_next = self.lstm_q(s_apriori, (h_0,c_0))
                estQ = self.fc_out_q(Fn.relu(self.h_q_next))
                outtxt4 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'estQ: shape and value' + str(estQ.shape) + ' ' + str(estQ) + '\n'
                fid4 = open(self.file4, 'a')
                fid4.write(outtxt4)
                fid4.close()
                #print("estQ: shape and value", estQ.shape, estQ)

                self.h_r_next, self.c_r_next = self.lstm_r(final_z, (h_0,c_0))
                estR = self.fc_out_q(Fn.relu(self.h_r_next))
                outtxt5 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'estR: shape and value' + str(estR.shape) + ' ' + str(estR) + '\n'
                fid5 = open(self.file5, 'a')
                fid5.write(outtxt4)
                fid5.close()
                #print("estR: shape and value", estR.shape, estR)

            elif k >1:
                
                self.h_next, self.c_next = self.lstm_f(final_input, (self.h_next, self.c_next))
                outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'h_next: shape and value: ' + str(
                    self.h_next.shape) + ' ' + str(self.h_next) + '\n'
                fid1 = open(self.file1, 'a')
                fid1.write(outtxt1)
                fid1.close()

                outtxt2 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'c_next: shape and value: ' + str(
                    self.c_next.shape) + ' ' + str(self.c_next) + '\n'
                fid2 = open(self.file2, 'a')
                fid2.write(outtxt2)
                fid2.close()

                s_apriori = Fn.relu(self.fc_out_f(Fn.relu(self.h_next)))
                outtxt3 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 's_apriori: shape and value' + str(
                    s_apriori.shape) + ' ' + str(s_apriori) + '\n'
                fid3 = open(self.file3, 'a')
                fid3.write(outtxt3)
                fid3.close()
                #print("S_apriori: shape and value", s_apriori.shape, s_apriori)

                self.h_q_next, self.c_q_next = self.lstm_q(s_apriori, (self.h_q_next, self.c_q_next))
                estQ = self.fc_out_q(Fn.relu(self.h_q_next))
                #print("estQ: shape and value", estQ.shape, estQ)
                outtxt4 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'estQ: shape and value' + str(
                    estQ.shape) + ' ' + str(estQ) + '\n'
                fid4 = open(self.file4, 'a')
                fid4.write(outtxt4)
                fid4.close()
                self.h_r_next, self.c_r_next = self.lstm_q(final_z, (self.h_r_next, self.c_r_next))
                estR = self.fc_out_q(Fn.relu(self.h_r_next))

                #print("estR: shape and value", estR.shape, estR)
                outtxt5 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'estR: shape and value' + str(
                    estR.shape) + ' ' + str(estR) + '\n'
                fid5 = open(self.file5, 'a')
                fid5.write(outtxt5)
                fid5.close()
            #print("Input before calling jacobian", final_input.type)

            F = self.get_jacobian(self.lstm_f, self.fc_out_f, final_input)

            fid7 = open(self.file7, 'a')
            np.savetxt(fid7, F, delimiter=',')
            fid7.close()
            #outtxt7 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'F: shape and value' + str(F.shape) + ' ' + str(estQ) + '\n'
            I = torch.eye(3) #identity matrix of suitale shape

            #PCurr = torch.from_numpy(np.array(PCurr))
            #PCurr = PCurr.to(dtype=torch.float32)
            #PCurr = Variable(PCurr, requires_grad=True)

            #print("PCurr: shape and value", PCurr.shape, PCurr)
            outtxt6 = 'AGV: ' + str(self.ownNo) + ' ' + 'k: ' + str(k) + ' ' + 'P_curr: shape and value' + str(PCurr.shape) + ' ' + str(PCurr) + '\n'
            fid6 = open(self.file6, 'a')
            fid6.write(outtxt6)
            fid6.close()

            #print("F: shape and value", F.shape, F)
            #print("PCurr er shape", PCurr.shape)
            sNext, PNext = self.kfObj(k, s_apriori, F, estQ, estR, PCurr, final_z, I)

            #fid8 = open(self.file8, 'a')
            #np.savetxt(fid8, sNext, delimiter=',')
            #fid8.close()

            #fid9 = open(self.file9, 'a')
            #np.savetxt(fid9, PNext, delimiter= ',')
            #fid9.close()

            #print("sNext: shape and value", sNext.shape, sNext)

            #print("PNext: shape and value", PNext.shape, PNext)

            #s

            #np.savetxt('estimate.xls', sNext, delimiter=',')
            return sNext, PNext
        

    






