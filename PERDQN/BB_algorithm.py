import numpy as np
from pulp import *
from utils import *
import torch
#from comparison_algorithm1 import *

def BB_alg(num_user, com_pow_edge, com_pow_list,  alpha_list, request_moment):

# 创建最大化问题
    prob = LpProblem("GQAP", LpMaximize)

    # 定义变量
    n_user = num_user
    n_action = 2


    x = LpVariable.dicts("x", [(i, k) for i in range(n_user) for k in range(n_action)], cat='Binary')
    y = LpVariable.dicts("y", [(i, k, j, l) for i in range(n_user) for k in range(n_action) for j in range(n_user) for l in range(n_action)],
                     cat='Binary')

    # 假设已有的距离矩阵和流量矩阵
    cost_matrix = cost_matrix_generation(num_user, com_pow_edge, com_pow_list,  alpha_list, request_moment)
    #print("cost_matrix",cost_matrix)

    prob += lpSum(cost_matrix[i][k][j][l] * y[i, k, j, l] for i in range(n_user) for k in range(n_action) for j in range(n_user) for l in range(n_action))

    # 添加约束条件示例：总选择的 `x[i, j]` 数量不能超过 10
    prob += lpSum(x[i, 0] for i in range(n_user)) <= 18
    for i in range(n_user):
        prob += lpSum(x[i, j] for j in range(n_action)) == 1

    for i in range(n_user):
        for k in range(n_action):
            for j in range(n_user):
                for l in range(n_action):
                    prob += y[i, k, j, l] <= x[i, k]
                    prob += y[i, k, j, l] <= x[j, l]
                    prob += y[i, k, j, l] >= x[i, k] + x[j, l] - 1

    # 求解问题
    prob.solve()

    action_list_1 = []
    action_list_2 = []
    for i in range(n_user):
        #for j in range(2):
        action_list_1.append(int(x[i, 0].value()))
        action_list_2.append(int(x[i, 1].value()))
    #print("action1", action_list_1)

        #print(f"x[{i},{j}] = {x[i, j].value()}")

    # ggg = fitness(action_list_1, request_moment, alpha_list, com_pow_list, com_pow_edge)
    # print("ggg",ggg)


    return 0.1*value(prob.objective)*(1/num_user)

if __name__ == '__main__':
    num_GPU = 16
    num_user = 20
    np.random.seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    computing_power_profile = 0.001 * np.load("../computing_power_profile.npy")
    profile_range = np.arange(3, computing_power_profile.shape[0])
    weights = np.linspace(1, 2, len(profile_range))  # 权重线性增加
    weights = weights / weights.sum()  # 归一化权重，使得总和为1
    chosen_indices = np.random.choice(profile_range, size=num_user, p=weights, replace=True)
    chosen_indices = np.sort(chosen_indices)
    computing_power_list = computing_power_profile[chosen_indices, :]
    com_pow_edge = (1 / num_GPU) * computing_power_profile[0]
    alpha_list = np.array([alpha_determination_with_bias(x, com_pow_edge) for x in computing_power_list])
    request_moment = np.sort(np.random.uniform(0, 1, size=num_user))

    AA = BB_alg(num_user, com_pow_edge, computing_power_list,  alpha_list, request_moment)
    print("AA", AA)

# BB = DQN_method(computing_power_list, alpha_list, request_moment, max_batchsize, com_pow_edge)
# print("BB", BB)