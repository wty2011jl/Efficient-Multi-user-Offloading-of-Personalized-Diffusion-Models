import random

import torch
import pickle
from tqdm.auto import tqdm
from DQN import *
from PER_DQN import *
from SAC import *
from environment import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import scipy.io as scio
#from inner_sac import Inner_SAC

def add_after_nth_element(lst, N):
    if N < len(lst) - 1:  # 确保N是有效索引且后面有元素
        lst[N] += sum(lst[N+1:])  # 将后面的元素和加到第N个元素上
    return lst[:N+1]   # 超过会返回所有元素 不会报错

def moving_average(data, window_size = 100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def setup_seed(seed):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+1)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



loss1 = 0
loss2 = 0
loss3 = 0

MAX_EPISODES = 30000

BATCH_SIZE = 128#128                                # 样本数量
MEMORY_CAPACITY = 400000

TARGET_REPLACE_ITER = 2000
EPSILON = 0.5
TAU = 4
LR = 0.0001
GAMMA = 1

Round = 4


max_batchsize = 18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

computing_power_profile = 0.001 * np.load("computing_power_profile.npy") #考虑8种类别的显卡 按照由好到差排序  0.1 * 用于单位转化  = s
#com_pow_edge =  0.0625 * computing_power_profile[0]   # 10个GPU #0.0625

for p in range(2, Round):

    #num_request = 4 + 4*p
    num_com = 4 + 4*p
    num_com =16

    for k in range (5):
        num_request = 4 + 4 * k
        num_request =4

        score_history = []
        score_Accuracy_history = []
        score_Latency_history = []
        loss = []

        agent = PER_DQN(MEMORY_CAPACITY, TARGET_REPLACE_ITER, BATCH_SIZE, EPSILON, LR, GAMMA, TAU)



        #num_request = 20#np.clip(np.random.poisson(mean_num_user), 1, max_num_user)

        # print("num", num_request)

        # computing_power_list = np.array(
        #     [random.choice(computing_power_profile[3:, :]) for _ in range(num_request)])



        for j in tqdm(range(MAX_EPISODES), desc=f"Training process", colour="#00ff00"):
            # random.seed(6)
            # np.random.seed(66)
            # task_id = random.randint(1000, 3000)
            np.random.seed(1234)

            #array = np.array([4, 8, 12, 16, 20])
            #num_request = np.random.choice(array)
            #num_request = np.random.randint(4, 21)
            #num_request = 12



            com_pow_edge = (1 / num_com) * computing_power_profile[0]  # 10个GPU #0.0625

            profile_range = np.arange(3, computing_power_profile.shape[0])

            weights = np.linspace(1, 2, len(profile_range))  # 权重线性增加
            weights = weights / weights.sum()  # 归一化权重，使得总和为1

            # 进行加权随机抽样，size=N表示抽取N个数字
            chosen_indices = np.random.choice(profile_range, size=num_request, p=weights, replace=True)

            # Step 2: 随机选择这些索引
            # chosen_indices = np.array([random.choice(profile_range) for _ in range(num_request)])
            # print("chosen_indices", chosen_indices)
            # chosen_indices = np.array([10,   9,  7, 3,   9,  6,   5,  8, 4, 10, 10, 9,   2, 8, 9, 7, 10, 6, 8,  10 ])
            chosen_indices = np.sort(chosen_indices)  # [::-1]
            print("chosen_indices", chosen_indices)

            # Step 3: 根据选定的索引提取相应的 computing_power_profile 中的值
            computing_power_list = computing_power_profile[chosen_indices, :]

            alpha_list = np.array([alpha_determination_with_bias(x, com_pow_edge) for x in computing_power_list])
            request_moment = np.sort(np.random.uniform(0, 1, size=num_request))


            num_to_assigned = num_request

            A_t_history = []
            L_t_history = []
            losshistory =[]

            episode_reward_sum = 0
            env = Env(com_pow_edge)

            s_l, s_g, done = env.reset(computing_power_list,  alpha_list, request_moment, max_batchsize, com_pow_edge)

            list_s_l = []
            list_s_g = []
            list_action = []
            list_s_l_ = []
            list_s_g_ = []
            list_done = []

            #edge_index = generate_adjacency_matrix(num_request + 1)

            for k in range(num_request):  #max_step

                if done == True:
                    break

                #list_edge.append(edge_index)
                # if  k == 0 or k ==28:
                #     action = 1#agent.choose_action(x, edge_index)
                # else:
                #     action = 0
                #
                #action = 1
                #print("action", action)
                ssss = s_l.copy()
                list_s_l.append(ssss)
                ddddd = s_g.copy()

                list_s_g.append(ddddd)


                action = agent.choose_action(s_l, s_g)



                s_l_, s_g_, done = env.step(action)
                llll = s_l_.copy()
                gggg = s_g_.copy()

                num_to_assigned = num_to_assigned - 1

                list_s_l_.append(llll)
                list_s_g_.append(gggg)
                #list_edge_.append(edge_index)
                list_action.append(action)
                list_done.append(done)

                s_l = s_l_.copy()
                s_g = s_g_.copy()

                if agent.memory1_counter > 5000:  # 如果累计的transition数量超过了记忆库的固定容量1000
                    loss1= agent.learn()
                    losshistory.append(loss1.cpu().detach().numpy())

            if j > 500:
                agent.EPSILON = max(agent.EPSILON - EPSILON / (MAX_EPISODES - 20000), 0.001)#* 0.99
            if j > 1500:
                agent.TAU = max(agent.TAU - TAU / (MAX_EPISODES - 8000), 0.01) #* 0.999

            kkk = len(list_action)

            if len(list_action) < num_request:
                list_action = list_action + [0] * (num_request - len(list_action))

            list_reward, PAI, latency = env.episode_reward(list_action)
            list_reward = add_after_nth_element(list_reward, len(list_action))

            for s_l, s_g,  action, reward, s_l_, s_g_,  done in zip(list_s_l, list_s_g,  list_action, list_reward, list_s_l_, list_s_g_,  list_done):
                agent.store_transition(s_l, s_g,  action, reward, s_l_, s_g_,  done)

            score_history.append(np.mean(list_reward))
            Reward = np.mean(list_reward)
            ACC = np.mean(PAI)
            score_Accuracy_history.append(ACC)
            LAT = np.mean(latency)
            score_Latency_history.append(np.sum(LAT))
            lossss = np.mean(losshistory)
            loss.append(lossss)

            print("\n<<<<<<<<<Round:", p, "Episode:", j, "Reward", Reward, "accuracy", ACC, "latency",
              LAT, "action", kkk)
            print("loss", lossss, "explore", agent.EPSILON, "soft_tau", agent.TAU, "memoery", agent.memory1.alpha,  agent.memory1.beta,  "memoery2", agent.memory2.alpha, agent.memory2.beta)
            print("action list", list_action)

            if j>0 and (j+1) % 10000 == 0:

                # print("score", score_history)
                # np.save("MMMM/%d/%d/score_history.npy" %(num_com,num_request),  score_history)
                # np.save("MMMM/%d/%d/E_t_history.npy" %(num_com,num_request), score_Accuracy_history)
                # np.save("MMMM/%d/%d/Latency_history.npy" %(num_com,num_request), score_Latency_history)  # 40000

                # with open('MMMM/%d/%d/memory1.pkl' %(num_com,num_request), 'wb') as f:
                #     pickle.dump(agent.memory1, f)
                # with open('MMMM/%d/%d/memory2.pkl' %(num_com,num_request), 'wb') as f:
                #     pickle.dump(agent.memory2, f)
                #
                # torch.save(agent.eval_net, "MMMM/%d/%d/hat_eval_net" %(num_com,num_request))
                # torch.save(agent.target_net, "MMMM/%d/%d/hat_target_net" %(num_com,num_request))

                x0 = score_history
                x1 = score_Accuracy_history
                x2 = score_Latency_history
                x3 = loss

                number = list(range(0, len(x2)))
                fig, ax = plt.subplots(4, 1)
                ax[0].plot(number, x0, color='y', linewidth='0.6')
                ax[1].plot(number, x1, linewidth='0.6')
                ax[2].plot(number, x2, linewidth='0.6')
                ax[3].plot(number, x3, linewidth='0.6')

                plt.savefig('SCOM_SUN_SUC/%d/%d/my_plot.png' %(num_com,num_request), dpi=100, bbox_inches='tight')


