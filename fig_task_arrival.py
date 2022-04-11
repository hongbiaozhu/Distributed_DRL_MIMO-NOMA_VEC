import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from mec_env_var import *
from helper import *
import tensorflow as tf
import tflearn
import ipdb as pdb
import time
from matplotlib.font_manager import FontProperties  

font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=15) 
font2 = FontProperties(fname="C:/Windows/Fonts/times.ttf", size=15)
# rate_list = [2.5,3,3.5,4,4.5,5,5.5,6]
rate_list = [2.5,3,3.5,4,4.5,5]

def long_term_disc_reward(set):
    r=0
    gamma=0.99
    for i in range(0,set.shape[0]):
        r = r + gamma*set[i]
    return r

def output_avg(dir,arr):
    arr_num = 'arr_' + str(arr)
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList[:]:
        path = dir_path + name
        res = np.load(path)
        temp_rs = np.array(res[arr_num])
        avg_rs.append(temp_rs)
    avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
    return avg_rs

reward_ddpg = []
reward_local = []
reward_offload = []

power_ddpg = []
power_local = []
power_offload = []

buffer_ddpg = []
buffer_local = []
buffer_offload = []

for rate in rate_list:

    reward_ddpg.append(long_term_disc_reward(output_avg('test_M_ddpg_sigma0_02_rate' + str(rate) + '_lane2/step_result/',0)))
    reward_local.append(long_term_disc_reward(output_avg('test_M_GD_Local_lane2_rate_' + str(rate) + '/step_result/',0)))
    reward_offload.append(long_term_disc_reward(output_avg('test_M_GD_Offload_lane2_rate_' + str(rate) + '/step_result/',0)))

    power_ddpg.append(np.mean(output_avg('test_M_ddpg_sigma0_02_rate' + str(rate) + '_lane2/',1), axis=0))
    power_local.append(np.mean(output_avg('test_M_GD_Local_lane2_rate_' + str(rate) + '/',1), axis=0))
    power_offload.append(np.mean(output_avg('test_M_GD_Offload_lane2_rate_' + str(rate) + '/',1), axis=0))

    buffer_ddpg.append(np.mean(output_avg('test_M_ddpg_sigma0_02_rate' + str(rate) + '_lane2/',2), axis=0))
    buffer_local.append(np.mean(output_avg('test_M_GD_Local_lane2_rate_' + str(rate) + '/',2), axis=0))
    buffer_offload.append(np.mean(output_avg('test_M_GD_Offload_lane2_rate_' + str(rate) + '/',2), axis=0))

fig = plt.figure(figsize=(6*1.25, 4.5*1.25))

plt.plot(rate_list, reward_ddpg, color='#1f77b4', label='最优策略', marker='.')
plt.plot(rate_list,reward_local, color='salmon', label='本地贪婪', marker='*')
plt.plot(rate_list,reward_offload, color='darkred',label='卸载贪婪', marker='+')
plt.ylabel('长期折扣奖励',fontproperties=font)

# plt.plot(rate_list, power_ddpg, color='#1f77b4', label='最优策略', marker='.')
# plt.plot(rate_list, power_local, color='salmon', label='本地贪婪', marker='*')
# plt.plot(rate_list, power_offload, color='darkred',label='卸载贪婪', marker='+')
# plt.ylabel('平均功耗',fontproperties=font)


# plt.plot(rate_list, buffer_ddpg, color='#1f77b4', label='最优策略', marker='.')
# plt.plot(rate_list, buffer_local, color='salmon', label='本地贪婪', marker='*')
# plt.plot(rate_list, buffer_offload, color='darkred',label='卸载贪婪', marker='+')
# plt.ylabel('平均计算缓冲长度',fontproperties=font)


plt.grid(linestyle=':')
plt.xlabel('平均包到达率',fontproperties=font)
plt.legend(prop=font)
plt.show()