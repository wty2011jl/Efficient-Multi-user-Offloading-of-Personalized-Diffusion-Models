import numpy as np
from utils import *
import sys

def log_scale_reward(reward):
    return np.sign(reward) * np.log(1 + abs(reward))

class Env():
    def __init__(self,  com_pow_edge, max_bandwidth = 100*1e6, batch_capacity = 30, min_split_point = 80, max_split_point = 200, eta = 10, K = 1, prompt_size = 216, intermediate_size = 4.4e6, max_user = 20 ): #bit
        self.batch_capacity = batch_capacity
        self.max_bandwidth = max_bandwidth

        self.min_split_point = min_split_point
        self.max_split_point = max_split_point
        self.eta = eta  # spectral efficiency
        self.com_pow_edge = com_pow_edge
        self.K = K  # 每个up

        self.prompt_size = prompt_size
        self.intermediate_size = intermediate_size
        self.max_user = max_user


    def reset(self, com_pow_list,  alpha_list, request_moment, max_batchsize, edge_com_pow):

        self.alpha_list = alpha_list
        self.num_user = len(self.alpha_list)
        self.com_pow_list = np.array(com_pow_list)
        self.edge_pow = edge_com_pow
        self.prompt_size_list = np.zeros(len(alpha_list)) + self.prompt_size
        self.intermediate_size_list = np.zeros(len(alpha_list)) + self.intermediate_size
        self.request_moment = request_moment
        self.max_batchsize = max_batchsize
        self.unassigned_count = len(alpha_list)
        self.grant_count = 0
        self.deny_count = 0

        self.step_count = 0

        self.split_point_list = [self.min_split_point] * self.num_user
        self.PAI_list = [PAI(x) for x in self.split_point_list]

        matrice_C = 0.01 * self.alpha_list#np.multiply(self.PAI_list, self.alpha_list)  # 使元素值大小差不多
        matrice_D_L = self.com_pow_list[:, 0] + self.com_pow_list[:, 1]
        matrice_D_R = (self.K - self.request_moment).reshape(self.num_user, 1)

        #global_feature = np.array([0.1 * self.max_batchsize, 0.1 * self.num_assigned, 0.1 * self.num_user, 100*self.edge_pow[0], 100*self.edge_pow[1]])
        local_feature = np.column_stack((matrice_C, matrice_D_L, matrice_D_R))
        zeros = np.zeros((local_feature.shape[0],1))
        zeros[0] = 1
        # local_feature= np.column_stack((zeros, local_feature))
        # self.state = np.row_stack((local_feature, global_feature))
        self.s_l  = np.column_stack((zeros, local_feature))

        self.s_l = pad_matrix(self.s_l, self.max_user)
            
        self.s_g = np.array([0.1 * self.max_batchsize, 0.1 * self.unassigned_count, 0.1 * self.grant_count, 0.1*self.deny_count, 100*self.edge_pow[0], 100*self.edge_pow[1]])
        self.done = False


        return  self.s_l, self.s_g, self.done

    def step(self, action):

        self.step_count = self.step_count + 1
        self.unassigned_count = self.unassigned_count - 1

        #print("state", self.state)

        #self.state = self.state[1:,:] #第一个用户分配完毕，待分配用户减一，状态矩阵减一
        if action == 1:  # grant
            self.max_batchsize = self.max_batchsize - 1
            self.grant_count = self.grant_count + 1
            self.s_l[0,0]  = 2
        else:
            self.deny_count = self.deny_count + 1
            self.s_l[0,0]  = 3

        unassigned_user = self.s_l[1:, :]
        assigned_user = self.s_l[0, :]
        self.s_l = np.vstack((unassigned_user, assigned_user))
        self.s_l[0,0] = 1

        # assigned_user = self.s_l[0, :]
        # inter_matrix = np.delete(self.state, 0, axis=0)
        # self.state = np.insert(inter_matrix, -1, assigned_user, axis=0)


        if self.max_batchsize == 0 or  self.step_count == self.num_user:
            self.done = True
            #self.max_batchsize = self.max_batchsize + 8e-10

        #self.num_assigned = self.num_assigned + 1

        self.s_g = np.array([0.1 * self.max_batchsize, 0.1 * self.unassigned_count, 0.1 * self.grant_count, 0.1*self.deny_count, 100*self.edge_pow[0], 100*self.edge_pow[1]])
        #print("self.state", self.s_l, self.done)

        return  self.s_l, self.s_g, self.done

    def episode_reward(self, action_list):

        #print("action_list", action_list)
        action_list = np.array(action_list)

        x1 = np.where(action_list == 1)
        user_grant = x1[0]
        x0 = np.where(action_list == 0)
        user_deny = x0[0]

        grant_count = len(user_grant)
        #print("grant_count", grant_count)
        deny_count = len(user_deny)
        #print("pending_count", pending_count)

        R_T_list = self.K - self.request_moment

        # grant
        if grant_count != 0:

            batch_grant = grant_count
            alpha_grant = self.alpha_list[user_grant]
            grant_com_pow = self.com_pow_list[user_grant, :]

            grant_split_point = np.array([solving_split_point(x, y, self.com_pow_edge, batch_grant) for x, y in
                                 zip(grant_com_pow, alpha_grant)])

            #print("grant_split_point",grant_split_point)

            # reward PAI
            pai_grant = np.squeeze(np.array([PAI(x) for x in grant_split_point]))

            # reward Latency
            denoising_latency_grant = self.com_pow_edge[0] * batch_grant + self.com_pow_edge[1]
            L_E_grant_list = (self.max_split_point - grant_split_point) * denoising_latency_grant

            L_T_grant_list =  (self.prompt_size_list[user_grant] + self.intermediate_size_list[
                user_grant]) * grant_count / self.max_bandwidth * self.eta
            L_L_grant_list = np.squeeze(grant_split_point)* (grant_com_pow[:, 0] + grant_com_pow[:, 1])

            L_grant = np.squeeze(L_E_grant_list) + L_T_grant_list + L_L_grant_list + R_T_list[user_grant]

            r_grant = alpha_grant * pai_grant #- L_grant

        else:
            r_grant = np.array([])
            L_grant = np.array([])
            pai_grant = np.array([])


        if deny_count != 0:

            alpha_deny = self.alpha_list[user_deny]
            deny_com_pow = self.com_pow_list[user_deny, :]

            pai_deny = np.squeeze(np.array([PAI(self.max_split_point)] * deny_count))

            L_deny = self.max_split_point * (deny_com_pow[:,0] + deny_com_pow[:,1]) + R_T_list[user_deny]

            r_deny = alpha_deny * pai_deny #- L_deny

        else:
            r_deny = np.array([])
            L_deny = np.array([])
            pai_deny = np.array([])



        pai_list = np.zeros(len(self.request_moment))
        pai_list[user_grant] = pai_grant
        pai_list[user_deny] = pai_deny

        latency_list = np.zeros(len(self.request_moment))
        latency_list[user_grant] = L_grant
        #print("L_grant", L_grant)
        latency_list[user_deny] = L_deny
        #print("L_deny", L_deny)

        reward_list = np.zeros(len(self.request_moment))
        reward_list[user_grant] = r_grant
        #print("r_grant", r_grant)
        reward_list[user_deny] = r_deny
        #print("r_deny", r_deny)
        #print(np.sum(latency_list))
        reward_list[-1] = reward_list[-1] - np.sum(latency_list)

        #reward_list =  [log_scale_reward(x) for x in reward_list]

        #print("reward_list", reward_list)
        reward_list = reward_list


        return reward_list * 0.1, pai_list, latency_list




