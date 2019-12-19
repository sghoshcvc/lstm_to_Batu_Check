import numpy as np
import torch as pt
#import os
#from numpy.linalg import inv
    #(Ep,tRegd,state,regNo,F,V,G,H,R,Y,PCurr,self.sObs)
class kalman():
    def __init__(self, reg_n):
        self.reg_n = reg_n
        #self.k = k
        #self.Kalman_gain = np.empty((2*self.reg_n+1))    
        #self.state_apriori = np.empty((2*self.reg_n+1))
        #self.next_estimate = np.empty((2*self.reg_n+1))
    def __call__(self, k, state_apriori, F, estQ, estR, P_prev, z_t, I):
       #Predict cycle
        first = pt.matmul(F,P_prev)
        second = pt.transpose(F, 0, 1)

        #print("P_prev er shape", P_prev.shape)
        #print("F er shape2", F.shape)

        #print("First: shape and value: ", first.shape, first)

        #print("Second: shape and value: ", second.shape, second)

        #print("estQ: shape and value: ", estQ.shape, estQ)

        matmul = pt.matmul(first,second)

        #print("matmul: shape and value: ", matmul.shape, matmul)
        P_apriori =  matmul + estQ
        #(1X3)
        #print("P_apriori: shape and value: ", P_apriori.shape, P_apriori)

        #Correct cycle
        S = P_apriori + estR

        #print("S: shape and value: ", S.shape, S)
        #if k == 1:
        inverse = 1/S
        #(1X3)
        #print("inverse: shape na dvalue", inverse.shape, inverse)
        #elif k > 1:
        #inverse = pt.inverse(S)

        Kalman_gain = P_apriori*inverse

        #print("Kalman_gain: shape and value: ", Kalman_gain.shape, Kalman_gain)
        #print("state_apriori: shape and value", state_apriori.shape, state_apriori)

        #print("z_t: shape and value", z_t.shape, z_t)
        #print("prod",  Kalman_gain*(z_t-state_apriori))
        next_estimate = state_apriori + Kalman_gain*(z_t-state_apriori)
       
    
        P_next = (I- Kalman_gain)*P_apriori
    

        return next_estimate, P_next


    #return tao_hat_next, P_next