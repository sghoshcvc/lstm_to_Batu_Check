import numpy as np
import matplotlib.pyplot as plt
import statistics as sc
import pandas as pd
def growingMean(splited_list):
    grow_Mean = sc.mean(splited_list)

    return grow_Mean

file_name1 = "/home/pragna/Documents/Documents/lstm-kf/Res_JET_2/dataJET.xlsx"
#pt, rt = np.loadtxt(file_name1)#, delimiter =', ', usecols =(0), unpack = True)


df = pd.read_excel(file_name1) # can also index sheet by name or fetch all sheets
#mylist = df['PTimeObs'].tolist()
print("Column headings:")
print(df.columns)
growing_Mean_Obs = []
growing_Mean_Est = []
p_t_Obs = df['PtimeObs']
p_t_Est = df['P_Time_Est']
#petalLength = df['Petal length']
for i in range(0, len(p_t_Obs)):
    temp1 = growingMean(p_t_Obs[:i+1])
    growing_Mean_Obs.append(temp1)
for i in range(0, len(p_t_Est)):
    temp2 = growingMean(p_t_Est[:i+1])
    growing_Mean_Est.append(temp2)
x = np.ones(len(p_t_Obs))
x = np.cumsum(x)

plt.plot(x, growing_Mean_Obs, 'r--', x, growing_Mean_Est, 'bs')
plt.show()