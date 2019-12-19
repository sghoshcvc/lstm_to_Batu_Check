import numpy as np
import pandas as pd
import datetime
from random import randint
import os
import time
from torch.autograd import Variable
import locale
import torch
from statistics import mean
locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
class utility():
    def __init__(self, rootDic, noAGV):
        self.costTasks= {}
        self.dirName = rootDic
        self.base1 = "relax_data"
        self.base2 = "Predecessor"
        self.base3 = "taskSequence"
        self.suffix = '.txt'
        self.ownNo = noAGV
        self.file1 = os.path.join(self.dirName, self.base1 + str(self.ownNo) + self.suffix)
        self.file2 = os.path.join(self.dirName, self.base2 + str(self.ownNo) + self.suffix)
        self.file_name = "Times.xlsx"
        self.sheet_name = "Full_set_times"
        self.d = {}
        self.P = {}
        self.s = {}
        #self.predList = []
        self.locAGV = {}
        self.zero_state = np.empty([1,3])
        self.zero_P = []
        self.adj = np.empty([13,13])
       # d = {}
        temp1 = []
        temp2 = []
        temp3 = []
        itrApprc = 0
        c1 = 0
        c2 = 0
        c3 = 0
        dfs = pd.read_excel(self.file_name, sheet_name=self.sheet_name, usecols="C, D")
        # print(dfs)
        for index, row in dfs.iterrows():
            # print("index", index)
            if row[0] == 'Operation':
                c1 = c1 + 1
                #print("c1", c1)
                #print("row[1]", row[1])
                # itrApprc= itrApprc + 1
                temp1.append(row[1])
            # end if
            self.d['performance'] = temp1
            if row[0] == 'Movement':
                c2 = c2 + 1
                #print("c2", c2)
                temp2.append(row[1])
            self.d['approach'] = temp2
            if row[0] == 'Logistics':
                c3 = c3 +1
                temp3.append(row[1])
            #end if
            self.d['return'] = temp3

            #            po = [('Mon', 6421), ('Tue', 6412), ('Wed', 12416), ('Thu', 23483), ('Fri', 8978), ('Sat', 7657),
             #     ('Sun', 6555)]

            # Generate dataframe from list and write to xlsx.
            pd.DataFrame(temp1).to_excel('PTimeData.xlsx', header=True, index=False)
            pd.DataFrame(temp2).to_excel('ATimeData.xlsx', header=True, index=False)
            pd.DataFrame(temp3).to_excel('RTimeData.xlsx', header=True, index=False)
            #dict1 = {"number of storage arrays": 45, "number of ports": 2390}

            #df1 = pd.DataFrame(data=self.d['performance'])
            #df1 = (df1.T)
            #print(df)
            #df1.to_excel('dataJET.xlsx', sheet_name='Performance_Time')

            # end if
        # end for
        #self.obsMean = 0
        #arr_txt = [x for x in os.listdir(self.dirName) if x.endswith("Data.txt")]
        #for f in arr_txt:
            #if f[0] == 'C':
            #f1=f.replace('C','m')#[:-4]
            #f1 = f1.replace('Data.txt', '')  # [:-4]
            #print(f1)
            #self.d[f1] = np.loadtxt(self.dirName+f)
            #print(self.d[f1])
    def readObs(self, lastTask):
        if (len(self.costTasks)>0):
            for k, e in self.costTasks.items():
                if k == lastTask:
                    costObs = e
                    costObs = [np.array(costObs)]
                    costObs = np.array(costObs)
                    print("Obs er shape", costObs)
        return costObs




    def storeObs(self, k, currTask, estCost):

        if k <=1:
            aTime = self.d['approach'][k]
            pTime = self.d['performance'][k]
            rTime = self.d['return'][k]
            cost = np.array([aTime, pTime, rTime])
            self.costTasks[currTask] = cost
        elif k >1 :
            aTime = estCost[0]
            pTime = estCost[1]
            rTime = estCost[2]
            cost = np.array([aTime, pTime, rTime])
            self.costTasks[currTask] = cost


    def getMeanLegacy(self, itr):
        obs = [] 

        obs.append(self.d['approach'][itr])
        obs.append(self.d['performance'][itr])
        obs.append(self.d['return'][itr])
        obs.append(self.d['approach'][itr])
        obs.append(self.d['performance'][itr])
        obs.append(self.d['return'][itr])
        obs.append(self.d['performance'][itr])
        obs.append(self.d['approach'][itr])
        #print("Obs", obs)
        self.obsMean = mean(obs)

    def cov(self, m, rowvar=True, inplace=False):
        '''Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        '''
        #if m.dim() > 2:
           # raise ValueError('m has more than 2 dimensions')
        #if m.dim() < 2:
            #m = m.view(1, -1)
        #if not rowvar and m.size(0) != 1:
            #m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        if inplace:
            m -= torch.mean(m, dim=1, keepdim=True)
        else:
            m = m - torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()

    def get_zero_state(self, k):
        gen = {k: sum(v) for k, v in self.d.items()}
        mean_Approach = gen['approach']/217
        mean_Performance = gen['performance']/211
        mean_Return = gen['return']/280
        self.zero_state[0][0] = mean_Approach#self.d['approach'][0]#mean_Approach
        self.zero_state[0][1] = mean_Performance#self.d['performance'][0]#mean_Performance
        self.zero_state[0][2] = mean_Return#self.d['return'][0]#mean_Return

        #self.getMeanLegacy(k)
        sMean = [self.d['approach'][0], self.d['performance'][0], self.d['return'][0]]#[self.obsMean, self.obsMean, self.obsMean]
        sMean = np.array([sMean])
        print("sMean : shape and value", sMean.shape, sMean)
        print("zero_state : shape and value", self.zero_state.shape, self.zero_state)
        #minus = self.zero_state - sMean
        #transPoseMinus = np.transpose(minus)
        #prodCovP = minus*transPoseMinus#np.dot()
    #rint("prodCovP", prodCovP)
        m = np.concatenate((sMean, self.zero_state), axis=0)
        print("m : shape and value", m.shape, m)
        Transpose_m = np.transpose(m)
        print("m transpose: shape and value", Transpose_m.shape, Transpose_m)
        mT = torch.from_numpy(np.array(Transpose_m))
        print("mT", mT)
        mT = mT.to(dtype=torch.float32)
        mT = Variable(mT, requires_grad=True)
        print("mT", mT)
        #print("mT.size(1)", m.size(1))
        self.zero_P = self.cov(mT)
         #= np.mean(prodCovP)
        print("P_0", self.zero_P)

    def get_state(self, reqdTask):
        PNext = self.P[reqdTask]
        sNext = self.s[reqdTask]
        return PNext, sNext
    def store_state(self, PNext, sNext, currTask):
        self.P[currTask] = PNext
        self.s[currTask] = sNext


    def readMap(self):
        AdjMap  = pd.read_excel("adjacencyMap.xls", header=None)
        self.adj = AdjMap.as_matrix()
        noV = len(self.adj)
        allV = list(range(1, (noV+1)))
        return allV
    def storeCurrLoc(self, ownNo, storeLoc):
    ##### this function designates the current location of the robot ###
    # ##later this function will get feed from LOcation_module #####
    ################################################################
        self.locAGV[ownNo] = storeLoc
        #return myCurrLoc


    def getCurrLoc(self, ownNo):
    ##### this function designates the current location of the robot ###
    # ##later this function will get feed from LOcation_module #####
    ################################################################
        myCurrLoc = self.locAGV[ownNo]
        return myCurrLoc

    def findNeighbor(self, u):
        listNeighbor = []
        #for  in self.adj[u-1][:]:
        n = [i for i,x in enumerate(self.adj[u-1][:]) if x == 1]
        for i in n:
            listNeighbor.append(i+1)
            #if i == 1:
             #   pos = 
              # listNeighbor.append(pos) 
        #listNeighbor = self.adj[u-1][:]
        return listNeighbor

    def relax(self, u, v, wObj, wt_u_v):
        outtxt1 = 'Before relax: ' + 'AGV: ' + str(self.ownNo) + ' ' + 'u: ' + str(u) + ' ' + 'v: ' + str(v) + ' ' + 'd_v[u]: ' + str(wObj.d_v[u]) + 'd_v[v]: ' + str(wObj.d_v[v]) + 'Weight: '+ str(wt_u_v) + 'Pi_v[v]: ' + str(wObj.pi_v[v]) + '\n'
        fid1 = open(self.file1, 'a')
        fid1.write(outtxt1)
        fid1.close()
        #print ("u, v, d_v[v], d_v[u], wt_u_v", u, v, wObj.d_v[v], wObj.d_v[u], wt_u_v)
        if (wObj.d_v[v] > (wObj.d_v[u] + wt_u_v)) and (wt_u_v != 0):
            wObj.d_v[v] = (wObj.d_v[u] + wt_u_v)
            wObj.pi_v[v] = u
        outtxt2 = 'After relax: ' +  'AGV: ' + str(self.ownNo) + ' ' + 'u: ' + str(u) + ' ' + 'v: ' + str(
            v) + ' ' + 'd_v[u]: ' + str(wObj.d_v[u]) + 'd_v[v]: ' + str(wObj.d_v[v]) + 'Weight: ' + str(
            wt_u_v) + 'Pi_v[v]: ' + str(wObj.pi_v[v]) + '\n'
        fid1 = open(self.file1, 'a')
        fid1.write(outtxt2)
        fid1.close()
    def findTaskPath(self, u, it):
        taskSeq = []
        print("Pi_v", it.pi_v)
        for k, e in it.pi_v.items():
            #print("e in pi_v", e)
            if e == u:
               
                nextT = k
                #print("nextT", nextT)
                taskSeq.append(nextT)
                u = nextT
        return taskSeq
    def findLenPredecessor(self, it, currNode):
        prev = 0
        t = currNode
        print("currNode in lenPredecessor: ", currNode)
        lenPrednodes = 0
        predList = []
        while (prev!=currNode):
            print("prev:", prev)
            predList.append(prev)
            if (it.pi_v[t]!='NIL'):
                prev = it.pi_v[t]
                lenPrednodes = lenPrednodes +1 #allPrev_nodes.append(prev)
                t = prev
            else:
                #del predList[:]# del a[:]
                # del a[:]
                # del a[:]

                break
            #end if
        #end while
        outtxt3 = 'AGV: ' + str(self.ownNo) + ' ' + 'currNode: ' + str(currNode) + ' ' + 'Predecessors: ' + str(predList) + 'lenPrednodes: ' + str(lenPrednodes) + '\n'
        fid2 = open(self.file2, 'a')
        fid2.write(outtxt3)
        fid2.close()
         #del a[:]
        return lenPrednodes
    def get_time(self):
        return str(datetime.datetime.now()).replace(":","-").replace(".","-").replace(" ","-")





