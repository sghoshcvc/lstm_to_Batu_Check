#Main routin 
#import io
#import pythonds.basic.stack
#import math
#import os
#from createMaps import createMaps
from AGV import AGV
#from addNewTT import newTT
#from newOntCreate import newOntCreate
#import threading
from multiprocessing import Process, Manager
from multiprocessing import Process,Queue
import sys
#import time

#def func(x):
#    
#    print(x*x)
#    time.sleep(5)
#    print(2*x)
if __name__ == "__main__":
    
        rootDic = sys.argv[1]
        totalAGVno = 1
        #regNo = 4
        AGVObj=[]
        q = Queue()
        #ontObjList = {}
        #mapNo = 1
        #mObj = createMaps(mapNo, rootDic)
            
        c = 1    
        while ( c <= totalAGVno):
            AGVObj.append(AGV(rootDic, c))
            #ontObj = newOntCreate(c,rootDic)
            #ontObjList[c] = ontObj

            c = c+1
        #print(ontObjList)
        #end while 
        #q.put(ontObjList)
        with Manager() as manager:
                #ontObjList = manager.dict(ontObjList)
                p1 = Process(target = AGVObj[0].scheduling)
                #p2 = Process(target = AGVObj[1].scheduling)
                #p3 = Process(target = AGVObj[2].scheduling)
                #p4 = Process(target = AGVObj[3].scheduling)
        #p5 = Process(target = ontology())
        
                p1.start()
                #p2.start()
                #p3.start()
                #p4.start()
        #p5.start()
#end main
                p1.join()
                #p2.join()
                #p3.join()
                #p4.join()
        #p5.join()
    
