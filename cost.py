#from bilinear import bilinear_model
from Kf import kalman
#from fetchObs import fetchOwnObs, fetchOtherObs
from statistics import mean
from collections import defaultdict#, OrderedDict
import os
import numpy as np
import pickle
#from obsMatrix import observationMean
from readObs import readObs
#from addNewTT import newTT
#import time
from addNewTT import addNewTT, fetchOtherObs
import time
class matrixForm(readObs):
    def __init__(self,rootDic,regN,ownNo):
        self.P = defaultdict(dict)
        #self.XObs = defaultdict(dict)
        #self.edge_cost = defaultdict(dict)
        self.s = defaultdict(dict)
        readObs.__init__(self)
        #newTT.__init__(self,ownNo,rootDic)
        self.kfObj = kalman(regN, ownNo)
        self.lstm_f = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.lstm_q = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.lstm_r = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)

        
        
                 #(self,regNo,inPCurr,seqNeigh,currNeighbor,currNode,tRegd,currentSource,bModObj)
    def findCost(self, it, ownNO, k, completedTask, currTask):
        #########################################################################
        ### finding cost for each task using LSTM        ###
        ### 3 LSTMs are there, LSTM_f is for state_apriori, LSTM_q ########
        ### and LSTM_r are for estimating Q and R
        #### F is the Jacoian of f (LSTM function) w.r.t s_apriori ######
        #########################################################################
            maxKey = max(completedTask)
            lastTask = completedTask(maxKey)
            PCurr = self.P[lastTask]
            F = J #jacoian 
            aTimeObs, prTimeObs, rTimeObs = self.observation(k)
            z_t = [aTimeObs, prTimeObs, rTimeObs]
            I = I #identity matrix of suitale shape 
            s_apriori = self.lstm_f(sCurr)
            estQ = self.lstm_q(sCurr)
            estR = self.lstm_q(z_t)
            sNext, PNext = self.kfObj(s_apriori, F, estQ, estR, PCurr, z_t, I)
            
            self.P[currTask] = PNext
            self.s[currTask] = sNext
            aTime = self.sNext[0]
            prTime= self.sNext[1]
            rTime = self.sNext[2]
           
            return aTime, prTime, rTime
    #def forward(self):
        
    def observation(self, k):
        if (neighbor_no == 1 or neighbor_no == 3 or neighbor_no == 5 or neighbor_no == 7):
            if (i == 230 or i == 231 or i == 232 or i == 233 or i == 234 or i == 259 or i == 260 or i == 261 or i == 262 or i == 263 or i == 288 or i == 289 or i == 290 or i == 291 or i ==292 or i == 317 or i == 318 or i == 319 or i == 320 or i == 321 or i == 344 or i == 345 or i == 346 or i == 347 or i == 348 or i == 367 or i == 368 or i == 369 or i == 370 or i == 371):  
                                                       
                meanData = np.mean(self.d['m17s'])
            if (i == 593 or i == 594 or i == 595 or i == 596 or i == 597 or i == 598 or i == 599 or i == 600 or i == 622 or i == 623 or i == 624 or i == 625 or i == 626 or i == 627 or i == 628 or i == 629 or i == 630 or i == 652 or i == 653 or i == 654 or i == 655 or i == 656 or i == 657 or i == 658 or i== 659 or i == 660 or i == 682 or i == 683 or i == 684 or i == 685 or i == 686 or i == 687 or i == 688 or i == 689 or i == 690 or i == 708 or i == 709 or i == 710 or i == 711 or i == 712 or i == 713 or i == 714 or i == 715 or i == 716 or i == 731 or i == 732 or i == 733 or i == 734 or i == 735 or i == 736 or i == 737 or i == 738 or i == 739 or i == 755 or i == 756 or i == 757 or i == 758 or i == 759 or i == 760 or i == 761 or i == 762 or i == 763 or i == 377 or i == 378 or i == 379 or i == 380 or i == 381 or i == 400 or i == 401 or i == 402 or i == 403 or i == 404 or i == 423 or i == 424 or i == 425 or i == 426 or i == 427):
            
                meanData = np.mean(self.d['m17sp'])
            else:
                meanData = np.mean(self.d['m17p'])
        elif (neighbor_no == 2 or neighbor_no == 4 or neighbor_no == 6 or neighbor_no == 8):
                if (i == 230 or i == 231 or i == 232 or i == 233 or i == 234 or i == 259 or i == 260 or i == 261 or i == 262 or i == 263 or i == 288 or i == 289 or i == 290 or i == 291 or i ==292 or i == 317 or i == 318 or i == 319 or i == 320 or i == 321 or i == 344 or i == 345 or i == 346 or i == 347 or i == 348 or i == 367 or i == 368 or i == 369 or i == 370 or i == 371):  
                                                     
                    meanData = np.mean(self.d['m24s'])
                if (i == 593 or i == 594 or i == 595 or i == 596 or i == 597 or i == 598 or i == 599 or i == 600 or i == 622 or i == 623 or i == 624 or i == 625 or i == 626 or i == 627 or i == 628 or i == 629 or i == 630 or i == 652 or i == 653 or i == 654 or i == 655 or i == 656 or i == 657 or i == 658 or i== 659 or i == 660 or i == 682 or i == 683 or i == 684 or i == 685 or i == 686 or i == 687 or i == 688 or i == 689 or i == 690 or i == 708 or i == 709 or i == 710 or i == 711 or i == 712 or i == 713 or i == 714 or i == 715 or i == 716 or i == 731 or i == 732 or i == 733 or i == 734 or i == 735 or i == 736 or i == 737 or i == 738 or i == 739 or i == 755 or i == 756 or i == 757 or i == 758 or i == 759 or i == 760 or i == 761 or i == 762 or i == 763 or i == 377 or i == 378 or i == 379 or i == 380 or i == 381 or i == 400 or i == 401 or i == 402 or i == 403 or i == 404 or i == 423 or i == 424 or i == 425 or i == 426 or i == 427):
               
                    meanData = np.mean(self.d['m24f'])
                else:
                    meanData = np.mean(self.d['m24p'])
                #end if 
        #end if 
        #self.fid1 = open(self.file1, 'a')
        return meanData
    def getMeanLegacy(self, itr):
        obs = [] 
#d['m17p'][0]                 #x[i,j,:] = d1                            
        obs.append(self.d['m17s'][itr])           
        obs.append(self.d['m17p'][itr])            
        obs.append(self.d['m17sp'][itr])            
        obs.append(self.d['m17p'][itr])                               
        obs.append(self.d['m24s'][itr])                
        obs.append(self.d['m24p'][itr])                
        obs.append(self.d['m24f'][itr])                
        obs.append(self.d['m24p'][itr])                
        obsMean = mean(obs)
        return obsMean
    
    def getLegacy(self, mObj, i, neighbor_no, itr):
#        rowNo = i
#        print("rowNo:", rowNo)
#        xx=mObj.neighbor_node_no[rowNo]
#        print("xx:", xx)
#        neighbor_no = (list(xx.keys())[list(xx.values()).index(dest)])
        #neighbor_no = np.where(mObj.neighbor[rowNo]==dest)[0]
        if (neighbor_no == 1 or neighbor_no == 3 or neighbor_no == 5 or neighbor_no == 7):
            #if (i == 230 or i == 231 or i == 232 or i == 233 or i == 234 or i == 259 or i == 260 or i == 261 or i == 262 or i == 263 or i == 288 or i == 289 or i == 290 or i == 291 or i ==292 or i == 317 or i == 318 or i == 319 or i == 320 or i == 321 or i == 344 or i == 345 or i == 346 or i == 347 or i == 348 or i == 367 or i == 368 or i == 369 or i == 370 or i == 371):
                                                 
            a=self.d['m17s'][itr]
            #else:
            b=self.d['m17p'][itr]
            #if (i == 593 or i == 594 or i == 595 or i == 596 or i == 597 or i == 598 or i == 599 or i == 600 or i == 622 or i == 623 or i == 624 or i == 625 or i == 626 or i == 627 or i == 628 or i == 629 or i == 630 or i == 652 or i == 653 or i == 654 or i == 655 or i == 656 or i == 657 or i == 658 or i== 659 or i == 660 or i == 682 or i == 683 or i == 684 or i == 685 or i == 686 or i == 687 or i == 688 or i == 689 or i == 690 or i == 708 or i == 709 or i == 710 or i == 711 or i == 712 or i == 713 or i == 714 or i == 715 or i == 716 or i == 731 or i == 732 or i == 733 or i == 734 or i == 735 or i == 736 or i == 737 or i == 738 or i == 739 or i == 755 or i == 756 or i == 757 or i == 758 or i == 759 or i == 760 or i == 761 or i == 762 or i == 763 or i == 377 or i == 378 or i == 379 or i == 380 or i == 381 or i == 400 or i == 401 or i == 402 or i == 403 or i == 404 or i == 423 or i == 424 or i == 425 or i == 426 or i == 427):
           
            c=self.d['m17sp'][itr]
            self.xObs = (a+b+c)/3
           # else:
                #self.xObs=self.d['m17p'][itr]
        elif (neighbor_no == 2 or neighbor_no == 4 or neighbor_no == 6 or neighbor_no == 8):
            #if (i == 230 or i == 231 or i == 232 or i == 233 or i == 234 or i == 259 or i == 260 or i == 261 or i == 262 or i == 263 or i == 288 or i == 289 or i == 290 or i == 291 or i ==292 or i == 317 or i == 318 or i == 319 or i == 320 or i == 321 or i == 344 or i == 345 or i == 346 or i == 347 or i == 348 or i == 367 or i == 368 or i == 369 or i == 370 or i == 371):
                                                       
            a=self.d['m24s'][itr]
            #else:
            b=self.d['m24p'][itr]
            #if (i == 593 or i == 594 or i == 595 or i == 596 or i == 597 or i == 598 or i == 599 or i == 600 or i == 622 or i == 623 or i == 624 or i == 625 or i == 626 or i == 627 or i == 628 or i == 629 or i == 630 or i == 652 or i == 653 or i == 654 or i == 655 or i == 656 or i == 657 or i == 658 or i== 659 or i == 660 or i == 682 or i == 683 or i == 684 or i == 685 or i == 686 or i == 687 or i == 688 or i == 689 or i == 690 or i == 708 or i == 709 or i == 710 or i == 711 or i == 712 or i == 713 or i == 714 or i == 715 or i == 716 or i == 731 or i == 732 or i == 733 or i == 734 or i == 735 or i == 736 or i == 737 or i == 738 or i == 739 or i == 755 or i == 756 or i == 757 or i == 758 or i == 759 or i == 760 or i == 761 or i == 762 or i == 763 or i == 377 or i == 378 or i == 379 or i == 380 or i == 381 or i == 400 or i == 401 or i == 402 or i == 403 or i == 404 or i == 423 or i == 424 or i == 425 or i == 426 or i == 427):
                
            c=self.d['m24f'][itr]
           # else:
            self.xObs=(a+b+c)/3
        #end if 
        
        #return obs
    def accRCY(self, estX, legX):
        perDiff = ((estX-legX)/legX)*100
        self.growPerDiff = self.growPerDiff + perDiff
    #end accRCY
    
    






