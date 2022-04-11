import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from helper import *
from matplotlib.font_manager import FontProperties  

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
	dir_path = dir
	fileList = os.listdir(dir_path) 
	fileList = [name for name in fileList if '.npz' in name]
	avg_rs = []
	for name in fileList:
		path = dir_path + name
		res = np.load(path)
		temp_rs = np.array(res['arr_2'])
		avg_rs.append(temp_rs)
	avg_rs = moving_average(np.mean(avg_rs, axis=0, keepdims=True)[0],10)
	return avg_rs

ddpg_buffer = output_avg('test_S_ddpg_sigma0_02_rate3_lane2/step_result/')
GD_local_buffer = output_avg('test_S_GD_local_lane2/step_result/')
GD_offload_buffer = output_avg('test_S_GD_Offload_lane2/step_result/')

# fig = plt.figure(figsize=(6, 4.5))
# plt.plot(range(ddpg_buffer.shape[0]), ddpg_buffer, color='#1f77b4', label='最优策略',lw=1 )
# plt.plot(range(GD_local_buffer.shape[0]), GD_local_buffer, color='salmon', label='本地贪婪策略',lw=1)
# plt.plot(range(GD_offload_buffer.shape[0]), GD_offload_buffer, color='darkred',label='卸载贪婪策略',lw=1)

# plt.grid(linestyle=':')
# plt.legend()
# plt.ylabel('计算任务缓存长度')
# plt.xlabel('时隙数')
# plt.show()
# fig.savefig('figs/buffer.eps', format='eps', dpi=1000)



font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=22) 
font1 = {'family' : 'SimSun',
'weight' : 'normal',
'size'   : 22,
}

figure, ax = plt.subplots(figsize=(7.5, 6))

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=22)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# plt.plot(range(step_op.shape[0]), step_op, color='#1f77b4', label="任务卸载功率", lw=1 )

plt.plot(range(ddpg_buffer.shape[0]), ddpg_buffer, color='0.25', label='最优策略',lw=1 )
plt.plot(range(GD_local_buffer.shape[0]), GD_local_buffer, color='0.5', label='本地贪婪策略',lw=1)
plt.plot(range(GD_offload_buffer.shape[0]), GD_offload_buffer, color='0.75',label='卸载贪婪策略',lw=1)
plt.legend(prop=font1)

plt.grid(linestyle=':')
plt.ylabel("计算任务缓存长度",fontproperties=font)
plt.xlabel("时隙数\n(b)",fontproperties=font)
plt.show()



