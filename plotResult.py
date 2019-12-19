import numpy as np
import pandas as pd
file_loc1 = "estimate.xls"
data1 = pd.read_excel(file_loc1)
file_loc2 = "observations.xls"
data2 = pd.read_excel(file_loc2)
df1 = pd.DataFrame(data1, columns= ['aTime'])
df2 = pd.DataFrame(data2, columns= ['aTime'])
df = pd.DataFrame(df1,df2)
