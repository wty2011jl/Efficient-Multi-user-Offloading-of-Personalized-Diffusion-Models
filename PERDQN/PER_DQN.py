import torch
from torch import nn
from torch_geometric.data import Data, Batch #自动reshape
from utils import *
from p_memory import *

import torch.nn.init as init

class MLP_Q_NET(nn.Module):
    def __init__(self, n_states = 21*6, n_actions =2, n_hidden=128):
        super(MLP_Q_NET, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_status = nn.Embedding(num_embeddings=4, embedding_dim=3)
        self.mlp1 = nn.Linear(n_states, 256).to(self.device)
        self.mlp2 = nn.Linear(256, 256).to(self.device)
        self.mlp3 = nn.Linear(256, 256).to(self.device)
        self.mlp4 = nn.Linear(256, n_actions).to(self.device)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x_l, x_g):  # batchsize * num_user * dim_feature
        x_g = torch.from_numpy(x_g).to(torch.float32).view(x_g.shape[0], -1).to(self.device)
        x_l = torch.from_numpy(x_l).to(torch.float32).to(self.device)
        #edge_index = torch.from_numpy(edge_index).to(torch.long).to(self.device)
        b_s = self.embedding_status(x_l[:,:,0].long())
        x_l = torch.cat((b_s, x_l[:,:, 1:]), dim=2)
        x_l = x_l.view(x_g.shape[0], -1)

        x = torch.cat((x_g, x_l), dim=1)
        x = self.mlp1(x)
        x = self.leaky_relu(x)
        x = self.mlp2(x)
        x = self.leaky_relu(x)
        x = self.mlp3(x)
        x = self.leaky_relu(x)
        x = self.mlp4(x)
        return x






class PER_DQN(object):
    def __init__(self, MEMORY_CAPACITY, TARGET_REPLACE_ITER, BATCH_SIZE, EPSILON, LR, GAMMA, TAU):  # 定义DQN的一系列属性

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        #self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.EPSILON = EPSILON
        self.TAU = TAU
        self.eval_net, self.target_net = MLP_Q_NET().to(self.device), MLP_Q_NET().to(self.device)
        self.eval_net = torch.load(r"Simulation\SCOM_SUN_SUC\16\4\hat_eval_net")
        #self.target_net = torch.load(r"Simulation\DCOM_DUN_DUC\hat_target_net_17")
        self.learn_step_counter = 0  # for target updating
        self.memory1_counter = 0  # for storing memory
        self.memory2_counter = 0  # for storing memory
        self.memory1 = ReplayTree(capacity=int(MEMORY_CAPACITY * 7/8)) # 初始化记忆库，一行代表一个transition
        self.memory2 = ReplayTree(capacity=int(MEMORY_CAPACITY * 1/8))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, weight_decay=0.0004)  # 使用Adam优化器 (输入为评估网络的参数和学习率) , weight_decay=0.0004
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.GAMMA = GAMMA

    def choose_action(self, s_l, s_g): # 定义动作选择函数 (x为状态)

        p = random.random()
        if p >= 0:#self.EPSILON:
            #print("retert", x.shape)
            #s = torch.from_numpy(x).to(torch.float32).reshape(1,x.shape[0], x.shape[1]).to(self.device)
            s_l = s_l.reshape(1, s_l.shape[0], s_l.shape[1])
            s_g = s_g[None, :]
            self.eval_net.eval()
            actions_value = self.eval_net.forward(s_l, s_g)    # 通过对评估网络输入状态x，前向传播获得动作值
            #print("actionvalue", actions_value)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]

            # softmax explore
            # q_values = self.eval_net.forward(s_l, s_g)  # Predict Q-values for current state
            # Q_VALUE = q_values[0].cpu().detach().numpy()
            # # Use Softmax Exploration to choose action based on Q-values
            # probabilities = softmax( Q_VALUE - np.max(Q_VALUE), tau=self.TAU)
            # action = np.random.choice([0, 1], p=probabilities)
        else:
            action = random.randint(0,1)
            #print("action", action)
        return action

    def store_transition(self, s_l, s_g,  a, r, s_l_, s_g_, done):

        transition = Data(s_l=s_l, s_g=s_g, action=a, reward=r, s_l_=s_l_, s_g_=s_g_,
                          done=done)

        s_l = s_l.reshape(1, s_l.shape[0], s_l.shape[1])
        s_g = s_g[None, :]


        s_l_ = s_l_.reshape(1, s_l_.shape[0], s_l_.shape[1])
        s_g_= s_g_[None, :]


        policy_val = self.eval_net(s_l, s_g).squeeze()[a]
        #print("policy_val", policy_val)
        target_val = self.target_net(s_l_, s_g_).squeeze()
        #print("target_val", target_val)


        if done:
            error = abs(policy_val - r)
            self.memory2.push(error, transition)
            self.memory2_counter += 1
        else:
            error = abs(policy_val - r - self.GAMMA * torch.max(target_val))
            self.memory1.push(error, transition)  # 添加经验和初始优先级
            self.memory1_counter += 1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1

        batch1, tree_idx1, is_weights1 = self.memory1.sample(int(self.BATCH_SIZE * 7 / 8))
        batch2, tree_idx2, is_weights2 = self.memory2.sample(int(self.BATCH_SIZE * 1 / 8))

        b_memory = batch1 + batch2

        # 在[0, 2000)内随机抽取32个数，可能会重复
        #b_memory = [self.memory[i] for i in sample_index]  # 抽取32个索引对应的32个transition，存入b_memory

        batch = Batch.from_data_list(b_memory)

        b_s_l = batch.s_l
        b_s_g = batch.s_g

        b_a = batch.action
        b_r = batch.reward
        b_s_l_ = batch.s_l_
        b_s_g_ = batch.s_g_

        b_dones = batch.done

        b_s_l = np.array(b_s_l)
        b_s_g = np.array(b_s_g)
        b_s_l_ = np.array(b_s_l_)
        b_s_g_ = np.array(b_s_g_)

        # b_edge = np.array(b_edge)
        # b_edge_ = np.array(b_edge_)


        #print("b_done", b_dones.dtype)

        b_a = torch.LongTensor(b_a).unsqueeze(1).to(self.device)
        #print("b_a", b_a.shape)
        b_dones = b_dones.int().unsqueeze(1).to(self.device)
        #print("b_dones", b_dones)
        b_r = b_r.to(torch.float32).unsqueeze(1).to(self.device)


        q_eval = self.eval_net.forward(b_s_l,b_s_g).gather(1, b_a)
        q_eval = q_eval.max(1)[0].view(self.BATCH_SIZE, 1)
        q_next = self.target_net.forward(b_s_l_,b_s_g_).detach()
        #print("b_r",b_r.shape)
        q_target = b_r + self.GAMMA * (1-b_dones)*q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        # print("q_target", q_target[0], "b_r", b_r[0], "q_eval", q_eval[0], "q_next", q_next[0])
        # print("q_target", q_target[-1], "b_r", b_r[-1], "q_eval", q_eval[-1], "q_next", q_next[-1])
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward() # 误差反向传播, 计算参数更新值
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=2.0)
        self.optimizer.step()   # 更新评估网络的所有参数

        abs_errors = torch.abs(q_eval - q_target).cpu().detach().numpy().squeeze()
        self.memory1.batch_update(tree_idx1, abs_errors[: len(batch1)])  # 更新经验的优先级
        self.memory2.batch_update(tree_idx2, abs_errors[len(batch1):])


        return loss


