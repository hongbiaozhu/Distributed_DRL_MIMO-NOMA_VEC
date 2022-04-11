import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  

font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=15) 
font2 = FontProperties(fname="C:/Windows/Fonts/times.ttf", size=15)
def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) 
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_0'])
		avg_rs.append(temp_rs)
	avg_rs = moving_average(np.mean(avg_rs, axis=0, keepdims=True)[0],30)
	return avg_rs

def long_term_disc_reward(set):
	r=0
	gamma=0.99
	for i in range(0,set.shape[0]):
		r = r + gamma*set[i]
	return r

ddpg_reward = output_avg('test_M_ddpg_sigma0_02_rate3_lane2/step_result/')
GD_local_reward = output_avg('test_M_GD_Local_lane2_rate_3/step_result/')
GD_offload_reward = output_avg('test_M_GD_Offload_lane2_rate_3/step_result/')


name = ["Polices"]
y1 = [long_term_disc_reward(ddpg_reward)]
y2 = [long_term_disc_reward(GD_local_reward)]
y3 = [long_term_disc_reward(GD_offload_reward)]


x = np.arange(len(name))
width = 0.25
fig = plt.figure(figsize=(6*1.25, 4.5*1.25))

plt.bar(x, y1,  width=width, label='最优策略',color='#1f77b4')
plt.bar(x + width, y2, width=width, label='本地贪婪', color='salmon', tick_label="")
plt.bar(x + 2 * width, y3, width=width, label='卸载贪婪', color='darkred')

# # 显示在图形上的值
# for a, b in zip(x,y1):
#     plt.text(a, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x,y2):
#     plt.text(a+width, b+0.1, b, ha='center', va='bottom')
# for a,b in zip(x, y3):
#     plt.text(a+2*width, b+0.1, b, ha='center', va='bottom')

plt.xticks()
# plt.grid(linestyle=':')

plt.ylabel('长期折扣奖励',fontproperties=font)
plt.xlabel('(c)',fontproperties=font2)
plt.legend(prop=font)
plt.show()
# fig.savefig('figs/buffer.eps', format='eps', dpi=1000)


# plt.show()
