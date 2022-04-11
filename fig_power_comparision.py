import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  

font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=15) 
font2 = FontProperties(fname="C:/Windows/Fonts/times.ttf", size=15)

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) 
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList[8:]:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_2'])
		avg_rs.append(temp_rs)
	avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
	return avg_rs


# ddpg_avg_power = np.mean(output_avg('test_S_ddpg_sigma0_02_rate3_lane2/'), axis=0)
# GD_local_avg_power = np.mean(output_avg('test_S_GD_Local_lane2/'), axis=0)
# GD_offload_avg_power = np.mean(output_avg('test_S_GD_Offload_lane2/'), axis=0)


ddpg_avg_power_m = np.mean(output_avg('test_M_ddpg_sigma0_02_rate3_lane2/'), axis=0)
GD_local_avg_power_m = np.mean(output_avg('test_M_GD_Local_lane2_rate_3/step_result/'), axis=0)
GD_offload_avg_power_m = np.mean(output_avg('test_M_GD_Offload_lane2_rate_3/'), axis=0)


# power = [ddpg_avg_power,GD_local_avg_power,GD_offload_avg_power]
power_M = [ddpg_avg_power_m, GD_local_avg_power_m, GD_offload_avg_power_m]
labels = ['DDPG', 'GD-Local', 'GD-Offload']

name = ["Policies"]
y1 = [ddpg_avg_power_m]
y2 = [GD_local_avg_power_m]
y3 = [GD_offload_avg_power_m]

print ((y2[0]-y1[0])/y2[0])
print ((y3[0]-y1[0])/y3[0])
print (y1,y2,y3)
x = np.arange(len(name))
width = 0.25

fig = plt.figure(figsize=(6*1.25, 4.5*1.25))


plt.bar(x, y1,  width=width, label='最优策略',color='#1f77b4')
plt.bar(x + width, y2, width=width, label='本地贪婪', color='salmon',tick_label="")
plt.bar(x + 2 * width, y3, width=width, label='卸载贪婪', color='darkred')


plt.xticks()
plt.ylabel('平均计算任务缓冲长度',fontproperties=font)
plt.xlabel('(a)',fontproperties=font2)

plt.grid(linestyle=':')

plt.legend(prop=font)
plt.show()