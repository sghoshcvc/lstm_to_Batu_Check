#from AGV import AGV
from initiliseDelta import initialD
from training import training
#from findWt import findWt
#from find_adj import find_adj
#from form_end_state import form_end_state
#from relax import relax
#from findLenPredecessor import findLenPredecessor
import numpy as np
#import pickle
#from findWt import findWt
import os
class K_nearestNeighbor():
    def __init__(self, rootDic, noAGV):
        self.dirName = rootDic
        #self.dirName = rootDic
        self.base1 = "nnTree"
        self.base2 = "cost"
        self.base3 = "taskSequence"
        #self.base4 = 'K_in_planning'
        self.suffix = '.txt'
        self.ownNo = noAGV

        self.B_curr = {} # 10 is the maximum number of neighbors one can have
        #self.B_dash_curr = {}
        #self.B_dash2_curr = {}
        self.lenT =0
        #self.k = 0
        self.stateDict = {}
        self.taskSequence = []
        self.endTask = 0
    def planning(self, currLoc, utilObj, params):
        u = currLoc
        trainObj = training(self.dirName, self.ownNo)
        allV = utilObj.readMap()
        it = initialD(allV, currLoc)
        update = 0
        while update == 0:
            #it.Q.remove(u)
            lenPred = utilObj.findLenPredecessor(it, u)
            k = lenPred +1
            self.B_curr[u] = utilObj.findNeighbor(u)
            f1 = os.path.join(self.dirName, self.base1 + str(self.ownNo) + self.suffix)
            fid1 = open(f1, 'a')
            outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'U: ' + str(u) + ' '+ 'B[u]' + '\n'
            fid1.write(outtxt1)
            fid1.close()
            fid1 = open(f1, 'a')
            np.savetxt(fid1, self.B_curr[u], delimiter=',')
            fid1.close()

            #print("Neighor of u", self.B_curr[u])
            if(len(self.B_curr[u])>0):
                for e in self.B_curr[u]:
                    if (k <=1):
                        prevTask = currLoc
                    elif k > 1:#      rootDic, k, u, utilObj, prevTask
                        prevTask = it.pi_v[u]
                    #print("K: ", k)
                    #print("PrevTask: ", prevTask)
                    estimatedCost = trainObj.computeCost(rootDic = self.dirName, noAGV = self.ownNo, k = k , u = u , utilObj = utilObj, prevTask= prevTask) #rootDic, noAGV, k, u, utilObj, prevTask
                    estimatedCost = estimatedCost.flatten()
                    f2 = os.path.join(self.dirName, self.base2 + str(self.ownNo) + self.suffix)
                    outtxt2 = 'AGV: '+ str(self.ownNo) + ' ' + 'U: ' + str(u) + ' ' + 'to' + ' ' + 'E: ' + str(e) + ' ' + str(estimatedCost[0]) + ' ' + str(estimatedCost[1]) + ' ' + str(estimatedCost[2]) + '\n'
                    fid2 = open(f2, 'a')
                    fid2.write(outtxt2)
                    fid2.close()
                    self.stateDict[k] = estimatedCost
                    utilObj.storeObs(k, e, estimatedCost)
                    sumCost = estimatedCost[0] + estimatedCost[1] + estimatedCost[2]
                    utilObj.relax(u, e, it, sumCost) #correct relax
                    self.B_curr[e] = utilObj.findNeighbor(e)
                    fid1 = open(f1, 'a')
                    outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'E: ' + str(e) + ' ' + 'B[e]' + '\n'
                    fid1.write(outtxt1)
                    fid1.close()
                    fid1 = open(f1, 'a')
                    np.savetxt(fid1, self.B_curr[e], delimiter=',')
                    fid1.close()
                    if (len(self.B_curr[e]) > 0):
                        for j in self.B_curr[e]:
                            if (j not in self.B_curr[u]) and (j != currLoc):
                                print("j: ", j)
                                lenPred = utilObj.findLenPredecessor(it, e)
                                k_dash = lenPred+1
                                prevTask = it.pi_v[e]
                                #print("K_dash: ", k_dash)
                                #print("PrevTask: ", prevTask)
                                estimatedCost = trainObj.computeCost(rootDic = self.dirName, noAGV = self.ownNo, k = k_dash , u = e , utilObj = utilObj, prevTask= prevTask)#(self.dirName, k, e, utilObj, prevTask)
                                estimatedCost = estimatedCost.flatten()
                                self.stateDict[k_dash] = estimatedCost
                                utilObj.storeObs(k_dash, j, estimatedCost)
                                f3 = os.path.join(self.dirName, self.base2 + str(self.ownNo) + self.suffix)
                                outtxt3 = 'AGV: ' + str(self.ownNo) + ' ' + 'E: ' + str(
                                    e) + ' ' + 'to' + ' ' + 'J: ' + str(j) + ' ' + str(estimatedCost[0]) + ' ' + str(
                                    estimatedCost[1]) + ' ' + str(estimatedCost[2]) + '\n'
                                fid3 = open(f3, 'a')
                                fid3.write(outtxt3)
                                fid3.close()
                                sumCost = estimatedCost[0] + estimatedCost[1] + estimatedCost[2]
                                utilObj.relax(e, j, it, sumCost)
                                self.B_curr[j] = utilObj.findNeighbor(j)
                                fid1 = open(f1, 'a')
                                outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'J: ' + str(j) + ' ' + 'B[j]' + '\n'
                                fid1.write(outtxt1)
                                fid1.close()
                                fid1 = open(f1, 'a')
                                np.savetxt(fid1, self.B_curr[j], delimiter=',')
                                fid1.close()
                                if (len(self.B_curr[j]) > 0):
                                    for h in self.B_curr[j]:
                                        if (h not in self.B_curr[u]) and (h not in self.B_curr[e]) and (h != currLoc):
                                            lenPred = utilObj.findLenPredecessor(it, h)
                                            k_ddash = lenPred + 1
                                            prevTask = it.pi_v[j]
                                            #print("K_ddash: ", k_ddash)
                                            #print("PrevTask: ", prevTask)
                                            estimatedCost = trainObj.computeCost(rootDic = self.dirName, noAGV = self.ownNo, k = k_ddash , u = j , utilObj = utilObj, prevTask= prevTask)#(self.dirName, k, j, utilObj, prevTask)
                                            estimatedCost = estimatedCost.flatten()
                                            self.stateDict[k_ddash] = estimatedCost
                                            utilObj.storeObs(k_ddash, h, estimatedCost)
                                            f4 = os.path.join(self.dirName, self.base2 + str(self.ownNo) + self.suffix)
                                            outtxt4 = 'AGV: ' + str(self.ownNo) + ' ' + 'J: ' + str(
                                                j) + ' ' + 'to' + ' ' + 'H: ' + str(h) + ' ' + str(
                                                estimatedCost[0]) + ' ' + str(
                                                estimatedCost[1]) + ' ' + str(estimatedCost[2]) + '\n'
                                            fid4 = open(f4, 'a')
                                            fid4.write(outtxt4)
                                            fid4.close()
                                            sumCost = estimatedCost[0] + estimatedCost[1] + estimatedCost[2]
                                            utilObj.relax(j, h, it, sumCost)
                                            self.B_curr[h] = utilObj.findNeighbor(h)
                                            fid1 = open(f1, 'a')
                                            outtxt1 = 'AGV: ' + str(self.ownNo) + ' ' + 'H: ' + str(h) + ' ' + 'B[h]' + '\n'
                                            fid1.write(outtxt1)
                                            fid1.close()
                                            fid1 = open(f1, 'a')
                                            np.savetxt(fid1, self.B_curr[h], delimiter=',')
                                            fid1.close()
                                            if (len(self.B_curr[h]) > 0):
                                                for l in self.B_curr[h]:
                                                    if (l not in self.B_curr[u]) and (l not in self.B_curr[e]) and ( l not in self.B_curr[j]) and (l != currLoc):
                                                        lenPred = utilObj.findLenPredecessor(it, h)
                                                        k_dddash = lenPred + 1
                                                        prevTask = it.pi_v[h]
                                                        #print("K_dddash: ", k_dddash)
                                                        #print("PrevTask: ", prevTask)
                                                        estimatedCost = trainObj.computeCost(rootDic = self.dirName, noAGV = self.ownNo, k = k_dddash , u = h , utilObj = utilObj, prevTask= prevTask)#(self.dirName, k, h, utilObj, prevTask)
                                                        estimatedCost = estimatedCost.flatten()
                                                        self.stateDict[k_dddash] = estimatedCost
                                                        utilObj.storeObs(k_dddash, l, estimatedCost)
                                                        f5 = os.path.join(self.dirName,
                                                                          self.base2 + str(self.ownNo) + self.suffix)
                                                        outtxt5 = 'AGV: ' + str(self.ownNo) + ' ' + 'H: ' + str(
                                                            h) + ' ' + 'to' + ' ' + 'L: ' + str(l) + ' ' + str(
                                                            estimatedCost[0]) + ' ' + str(
                                                            estimatedCost[1]) + ' ' + str(estimatedCost[2]) + '\n'
                                                        fid5 = open(f5, 'a')
                                                        fid5.write(outtxt5)
                                                        fid5.close()
                                                        sumCost = estimatedCost[0] + estimatedCost[1] + estimatedCost[2]
                                                        utilObj.relax(h, l, it, sumCost)
                                            else:
                                                break
                                else:
                                    break
                    else:
                        break
            else:
                break




            self.taskSequence = utilObj.findTaskPath(u, it)
            print("Tasksequence", self.taskSequence)
            f6 = os.path.join(self.dirName, self.base3 + str(self.ownNo) + self.suffix)
            fid6 = open(f6, 'a')
            # self.fid1.write(outtxt1)
            np.savetxt(fid6, self.taskSequence, delimiter=',')
            fid6.close()
            self.lenT = len(self.taskSequence)
            self.endTask = self.taskSequence[self.lenT-1]
            update = 1

            print("Len of stateDict", len(self.stateDict))
            return self.stateDict, self.lenT

