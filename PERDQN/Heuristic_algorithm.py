import numpy as np
import random
from utils import *
import torch

# 参数设置


# 随机生成距离矩阵和流量矩阵



# 个体适应度函数
# 初始种群生成
def initial_population(n):
    population_size = 100
    return [np.random.choice([0, 1], size=n) for _ in range(population_size)]


# 选择操作
def selection(population, fitness_scores):
    selected = random.choices(population, weights=fitness_scores, k=2)
    return selected[0], selected[1]


# 交叉操作
def crossover(parent1, parent2,n):
    crossover_point = random.randint(1, n - 1)
    child = np.zeros(n, dtype=int) #- 1
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    # for gene in parent2:
    #     if gene not in child:
    #         for i in range(n):
    #             if child[i] == -1:
    #                 child[i] = gene
    #                 break
    return child


# 变异操作
def mutation(child,n):
    mutation_rate = 0.1
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(n), 2)
        child[idx1], child[idx2] = child[idx2], child[idx1]


# 主遗传算法
def genetic_algorithm(num_user, request_moment, alpha_list, com_pow_list, com_pow_edge):
    population_size = 100  # 种群规模
      # 变异概率
    generations = 200  # 迭代次数
    population = initial_population(num_user)
    #print("proror", len(population))
    for _ in range(generations):
        fitness_scores = [fitness(individual,request_moment, alpha_list, com_pow_list, com_pow_edge) for individual in population]
        #print("fitness_scores", fitness_scores)
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitness_scores)
            child1 = crossover(parent1, parent2, num_user)
            child2 = crossover(parent2, parent1, num_user)
            mutation(child1,num_user)
            mutation(child2,num_user)
            new_population.extend([child1, child2])
        population = new_population

    PP = np.zeros(population_size)

    for i in range(population_size):
        PP[i] = fitness(population[i],request_moment, alpha_list, com_pow_list, com_pow_edge)
    #nppp = np.argmax(PP)
    #print("nppp", population[nppp])

    return  np.max(PP)

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
    #chosen_indices = np.sort(chosen_indices)
    computing_power_list = computing_power_profile[chosen_indices, :]
    com_pow_edge = (1 / num_GPU) * computing_power_profile[0]
    alpha_list = np.array([alpha_determination_with_bias(x, com_pow_edge) for x in computing_power_list])
    request_moment = np.sort(np.random.uniform(0, 1, size=num_user))

    # 运行遗传算法
    score = genetic_algorithm(num_user, request_moment, alpha_list, computing_power_list, com_pow_edge)

    print("Best Score:", score)
