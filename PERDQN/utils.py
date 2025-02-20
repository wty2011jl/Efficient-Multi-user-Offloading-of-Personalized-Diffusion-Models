import numpy as np
import random
from scipy.optimize import root_scalar
from torch_geometric.utils import dense_to_sparse

def PAI(x):
    """
    PAI function
    :param x: split point
    :return:
    """
    a = 0.0413
    b = 71.44
    #print("x", x)
    y = 1 / (1 + np.exp(-a*(x-b)))
    return y

def test(x):
    return PAI(x) * (1 - PAI(x))

# print("80", test(80))
# print("200", test(200))

def solving_split_point(com_pow_l,  alpha, com_pow_e, batchsize):
    a = 0.0413  # parameter for fitting curve
    #print("compow", com_pow_e)

    k_i = com_pow_l[0]
    h_i = com_pow_l[1]
    k_e = com_pow_e[0]
    h_e = com_pow_e[1]
    #print("k_e", k_e, "h_e", h_e)

    def PAI_derivative(x):
        return  alpha *  a * PAI(x) * (1 - PAI(x)) #
    latency_metric = k_i + h_i - k_e * batchsize - h_e

    # print ("latency_metric", k_i, h_e, latency_metric,batchsize, alpha)

    threthold = 4 * (latency_metric) / a
    # print ("threthgoud", threthold)
    # print("alpha", alpha, "threthold", threthold)




    if alpha <= threthold:

        solution = [80]
    else:


        #print("alpha", alpha)

        def equation(x, latency_metric):
            return PAI_derivative(x) - latency_metric

        lower_bound = 80
        upper_bound = 200

        AA = equation(lower_bound, latency_metric)
        BB = equation(upper_bound, latency_metric)
        #print( "AAA", AA, "BBB", BB )
        if AA*BB<0:
            result = root_scalar(equation, args=(latency_metric,), bracket=[lower_bound, upper_bound])
            solution = [result.root]
        elif AA + BB <0:
            solution = [80]
        else:
            solution = [200]
    #print("bcie", solution)
    return solution

def calculate_x_range(com_pow_l, com_pow_e, min_batchsize = 1, max_batchsize = 20):
    k_i = com_pow_l[0]
    h_i = com_pow_l[1]
    k_e = com_pow_e[0]
    h_e = com_pow_e[1]
    ymax = k_i + h_i - k_e * max_batchsize - h_e
    #print("k_i, h_i, k_e, max_batchsize,h_e", k_i, h_i, k_e, max_batchsize,h_e)
    ymin = k_i + h_i - k_e * min_batchsize - h_e
    #print("k_i, h_i, k_e, min_batchsize,h_e", k_i, h_i, k_e, min_batchsize, h_e)

    x_lower =  max(ymax / 0.01, 0)  # x > y / 0.01 (n* = 80)  alpha /4
    x_upper = 0.05*max(ymin / 0.0002,0)  # x < y / 0.0002 (n* = 200)       越小对时延约关注   0.08是给定的参数

    #print("x_lower, y_lower", x_lower, x_upper)

    return x_lower, x_upper



def alpha_determination_with_bias(com_pow_l, com_pow_e, alpha=2, beta=2):
    """
    使用 Beta 分布在 x 范围内生成一个数值，偏向中间区域。
    alpha, beta 控制分布形状，默认情况下 alpha=beta=2 倾向于中间。
    """
    x_range = calculate_x_range(com_pow_l, com_pow_e,)
    # if x_range:
    #     # 在 [0, 1] 区间生成 Beta 分布的随机数，alpha 和 beta 决定分布形状
    #     random_beta = np.random.beta(alpha, beta)
    #
    #     # 将 [0, 1] 区间的 Beta 随机数映射到 [x_lower, x_upper]
    #     x_random = x_range[0] + random_beta * (x_range[1] - x_range[0])
    #     return x_random
    # else:
    #     return None

    return np.random.uniform(x_range[0], x_range[1])


def generate_edge_index_GAT(num_nodes):
    """
    使用 numpy 生成一个 edge_index，其中：
    - 节点 0 和最后一个节点是目标节点。
    - 其他节点作为源节点连接到这两个目标节点。

    参数:
    - num_nodes: 图中的节点总数

    返回:
    - edge_index: numpy 数组，表示边的索引
    """

    # 0 号节点和最后一个节点是目标节点
    target_nodes = np.array([0, num_nodes - 1])

    # 源节点是从 1 到 num_nodes - 2 的节点
    source_nodes = np.arange(1, num_nodes - 1)

    # 创建边（源节点到目标节点）
    edges_to_target_0 = np.vstack([source_nodes, np.full(source_nodes.shape, target_nodes[0])])
    edges_to_target_last = np.vstack([source_nodes, np.full(source_nodes.shape, target_nodes[1])])

    # 拼接边，形成 edge_index
    edge_index = np.hstack([edges_to_target_0, edges_to_target_last])

    return edge_index


def generate_adjacency_matrix(N):
    # 创建一个 N x N 的零矩阵
    adjacency_matrix = np.zeros((N, N), dtype=int)

    # 前 N-1 个节点都与最后一个节点相连
    # for i in range(N - 1):
    #     adjacency_matrix[i, N - 1] = 1
    #     adjacency_matrix[N - 1, i] = 1  # 无向图对称

    # 第 2 个到第 N-2 个节点与第 1 个节点相连
    for i in range(1, N - 1):
        adjacency_matrix[i, N - 1] = 1
        adjacency_matrix[N - 1, i] = 1  # 无向图对称

    adjacency_matrix = np.array(np.nonzero(adjacency_matrix))

    return adjacency_matrix

#print("邻接矩阵", generate_adjacency_matrix(6))

def res_latency_generate_latency_list(init_ongoing_batch, min_0_1=1, max_0_1=10, low_0_1=0, high_0_1=1, low_1_5=1, high_1_5=8):
    """
    生成一个列表，其中包含随机数量的0到1之间的数（数量在min_0_1到max_0_1之间随机选择），
    其余数在1到5之间，打乱顺序。

    参数：
    init_ongoing_batch: 生成列表的总数。
    min_0_1: 0-1区间内最少包含的数的数量。
    max_0_1: 0-1区间内最多包含的数的数量。
    low_0_1: 0-1区间内的数的最小值（默认0）。
    high_0_1: 0-1区间内的数的最大值（默认1）。
    low_1_5: 其余数的最小值（默认1）。
    high_1_5: 其余数的最大值（默认5）。

    返回：
    打乱顺序的随机生成数的列表。
    """
    # 随机确定 0-1 之间的数的数量
    num_0_1 = np.random.randint(min_0_1, max_0_1 + 1)

    # 生成 num_0_1 个 0 到 1 之间的数
    latency_0_1 = np.random.uniform(low_0_1, high_0_1, size=num_0_1)

    # 生成其余的数，在 1 到 5 之间
    latency_1_5 = np.random.uniform(low_1_5, high_1_5, size=init_ongoing_batch)# - num_0_1

    # 合并两个列表
    latency_list = np.concatenate([latency_0_1, latency_1_5])

    # 打乱顺序
    np.random.shuffle(latency_list)

    return latency_1_5

def softmax(x, tau=1.0):
    """Compute the softmax of vector x with temperature tau."""
    x = np.array(x)
    exp_x = np.exp(x / tau)  # Apply temperature scaling
    return exp_x / np.sum(exp_x)


def pad_matrix(matrix, N):
    # 获取当前矩阵的行数和列数
    current_rows, cols = matrix.shape

    # 如果行数小于N，则补充行数
    if current_rows < N:
        # 创建需要补充的零矩阵 (N - current_rows 行, cols 列)
        padding = np.zeros((N - current_rows, cols))
        # 在现有矩阵下方拼接补充矩阵
        padded_matrix = np.vstack((matrix, padding))
        return padded_matrix
    else:
        # 如果行数已经大于或等于N，返回原矩阵
        return matrix