from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np
import cv2
import sys
import gymnasium as gym
import random
import ale_py
import time
from matplotlib import pyplot as plt
#游戏名
GAME = 'Pong-v5'
ACTIONS = 6          # Pong环境的动作数量 
GAMMA = 0.99         # 奖励折扣因子
OBSERVE = 20000      # 训练前观察步数1
EXPLORE = 1.0e6      # 探索总步数
FINAL_EPSILON = 0.01 # 最终探索率
INITIAL_EPSILON = 1.0 # 初始探索率
REPLAY_MEMORY = 100000 # 经验回放缓冲区大小
BATCH = 64           # 批处理大小
FRAME_PER_ACTION = 1 # 跳帧数 (Pong环境已内置4帧跳帧)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def layer_init(layer, std=np.sqrt(1),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# 图像预处理函数
def preprocess(observation):
    """
    对输入图像进行预处理：调整大小并二值化
    输入图像已经是灰度图 (grayscale_obs=True)
    """
    # 调整大小为84x84
    resized = cv2.resize(observation, (84, 84))
    # resized = observation
    # 使用阈值128进行二值化，保留关键游戏元素
    _, thresh = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
    # print(thresh)
    
    # 返回uint8类型的图像，值范围为[0, 255]
    return thresh.astype(np.uint8)

#定义经验回报类，完成数据的存储和采样
class Experience_Buffer():
    """经验回放缓冲区，用于存储和采样训练数据"""
    def __init__(self,buffer_size = REPLAY_MEMORY):
        self.buffer = []
        self.buffer_size = buffer_size
    def add_experience(self, experience):
        """添加新经验，当缓冲区满时移除最早的经验"""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # 移除最早的经验
        self.buffer.extend(experience)
    def sample(self,samples_num):
        sample_data = random.sample(self.buffer, samples_num)
        train_s = np.zeros((samples_num,4,84,84))
        train_a = np.zeros((samples_num,ACTIONS))
        train_r = np.zeros((samples_num,1))
        train_s_ = np.zeros((samples_num,4,84,84))
        train_terminal = np.zeros((samples_num,1))
        for i in range(samples_num):
            train_s[i] = sample_data[i][0]
            train_a[i] = sample_data[i][1]
            train_r[i] =sample_data[i][2]
            train_s_[i] = sample_data[i][3]
            train_terminal[i] = sample_data[i][4]
        
        # 归一化像素值到[0,1]范围，这对于神经网络处理很重要
        train_s = train_s.astype(np.float32) / 255.0
        train_s_ = train_s_.astype(np.float32) / 255.0
        
        # 转换为PyTorch张量并移到指定设备
        train_s = torch.as_tensor(train_s, dtype=torch.float32).to(device)
        train_a = torch.as_tensor(train_a, dtype=torch.float32).to(device)
        train_r = torch.as_tensor(train_r, dtype=torch.float32).to(device)
        train_s_ = torch.as_tensor(train_s_, dtype=torch.float32).to(device)
        train_terminal = torch.as_tensor(train_terminal, dtype=torch.float32).to(device)
        
        return train_s, train_a, train_r, train_s_, train_terminal

class Critic_Net(nn.Module):
    def __init__(self,action_dim):
        super(Critic_Net, self).__init__()
        self.layer_1 = layer_init(nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4,padding=2))
        self.pool_1 = nn.MaxPool2d(2,stride=2)
        self.layer_2 = layer_init(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2,padding=1))
        self.layer_3 = layer_init(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1))
        self.critic_fc1 = layer_init(nn.Linear(1600,512))
        self.critic_fc2 = layer_init(nn.Linear(512,action_dim))
        
    def forward(self,state_input):
        # 使用适当的normalized_shape参数来修复层归一化
        x = state_input
        x = F.relu(self.layer_1(x))
        x = self.pool_1(x)
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = x.view(-1, 1600)
        x = F.relu(self.critic_fc1(x))
        x = self.critic_fc2(x)
        return x

#DQN算法实现
class DQN():
    def __init__(self):
        self.tau = 0.01            # 软更新因子
        self.gamma = GAMMA         # 奖励折扣因子
        self.critic_lr = 0.0002    # 学习率
        self.batch_num = BATCH     # 批处理大小
        self.save_freq = 50000     # 模型保存频率
        self.target_update_freq = 3000  # 目标网络更新频率
        self.Q = Critic_Net(action_dim=ACTIONS).to(device)
        self.Q_ = Critic_Net(action_dim=ACTIONS).to(device)
        self.critic_optimizer = Adam(self.Q.parameters(), self.critic_lr)
        self.training_path = './models'
        self.critic_filename = 'deepQ_pong.pth'
        # 确保保存模型的目录存在
        os.makedirs(self.training_path, exist_ok=True)
        # 初始时复制在线网络到目标网络
        self.Q_.load_state_dict(self.Q.state_dict())
    def compute_loss_critic(self,obs,action,reward,obs_next,terminal):
        """计算批量损失函数，使用双Q学习方法
        参数:
            obs: 当前状态 [batch_size, 4, 84, 84] 
            action: 采取的动作(one-hot) [batch_size, ACTIONS]
            reward: 获得的奖励 [batch_size, 1]
            obs_next: 下一个状态 [batch_size, 4, 84, 84]
            terminal: 终止标志 [batch_size, 1]
            
        返回:
            critic_loss: MSE损失值
        """
        # 目标网络评估部分 - 无梯度
        with torch.no_grad():
            # 1. 使用online网络选择动作（保持一维）
            best_actions = torch.argmax(self.Q(obs_next), dim=1)
            
            # 2. 使用target网络评估价值（从一维变为二维）
            q_next = self.Q_(obs_next).gather(1, best_actions.unsqueeze(1))
            
            # 3. 计算目标Q值
            # reward和terminal形状为[batch_size, 1]
            q_target = reward + self.gamma * q_next * (1 - terminal)
            
            # 4. 统一转为一维张量用于最终的损失计算
            q_target = q_target.squeeze(1)  # [batch_size]
        
        # 5. 计算当前策略的Q值预测
        q_values = self.Q(obs)  # [batch_size, action_dim]
            
        # 6. 选出实际采取的动作的Q值（one-hot乘法并求和）
        q_pred = torch.sum(q_values * action, dim=1)  # [batch_size]
        
        # 7. 计算MSE损失
        critic_loss = F.mse_loss(q_pred, q_target)
        
        return critic_loss
        
    def epsilon_greedy(self,s_t,epsilon):
        # 归一化输入状态与批量处理保持一致
        s_t = s_t.astype(np.float32) / 255.0
        s_t = torch.as_tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 为Pong创建动作向量
        a_t = np.zeros([ACTIONS])
        with torch.no_grad():
            q = self.Q(s_t).cpu().numpy()
        amax = np.argmax(q[0])
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            a_t[amax] = 1
        else:
            a_t[random.randrange(ACTIONS)]=1
        return a_t
        
    def soft_update(self, target_net, eva_net, tau):
        for target_param, param in zip(target_net.parameters(), eva_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    def load_model(self, model_path, checkpoint):
        """从指定路径加载模型"""
        full_path = f"{model_path}.pth{checkpoint}"
        print(f"正在从 {full_path} 加载模型...")
        if os.path.exists(full_path):
            self.Q.load_state_dict(torch.load(full_path, map_location=device))
            self.Q_.load_state_dict(self.Q.state_dict())
            print("模型加载成功!")
            return True
        else:
            print(f"找不到模型文件: {full_path}")
            return False
    def test_Network(self):
        # 创建Atlantis游戏环境
        env = gym.make("ALE/Pong-v5",render_mode="human", obs_type="grayscale")
        env = gym.wrappers.AtariPreprocessing(env, frame_skip = FRAME_PER_ACTION, grayscale_obs= True, scale_obs=False, terminal_on_life_loss=True)
        
        # 获得第一个状态
        observation, info = env.reset()
        x_t = preprocess(observation)
        
        # 堆叠4帧作为初始状态
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        
        # 开始测试
        t = 0
        total_reward = 0
        
        while True:
            # 采用贪婪策略选择动作
            a_t = self.epsilon_greedy(s_t, epsilon = 0)
            a_t_index = np.argmax(a_t)
            
            # 执行动作，获取下一个状态
            observation, r_t, terminated, truncated, info = env.step(a_t_index)
            terminal = terminated or truncated
            
            total_reward += r_t
            
            # 处理新的观察
            x_t1 = preprocess(observation)
            x_t1 = np.reshape(x_t1, (1, 84, 84))
            
            # s_t1=[x_t1,xt,xt-1,xt-2]
            s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
            
            # 往前推进一步
            s_t = s_t1
            t += 1
            
            # 如果游戏结束或者达到一定步数，结束测试
            if terminal:
                print(f"游戏结束，总步数: {t}，总得分: {total_reward}")
                break
        
        env.close()
        return t, total_reward

if __name__ == "__main__":
    # 创建一个DQN智能体
    brain = DQN()
    brain.load_model('./models/deepQ_pong', 10000000)
    brain.Q.eval()  # 设置为评估模式
    brain.test_Network()



    '''
    # 可视化测试结果
    play_steps_max = []
    play_steps_min = []
    play_steps_avg = []

    play_scores_max = []
    play_scores_min = []
    play_scores_avg = []

    play_scores_per_step_max = []
    play_scores_per_step_min = []
    play_scores_per_step_avg = []

    for i in range(200):
        checkpoint = 50000 * (i + 1)
        brain.load_model('./models/deepQ_pong', checkpoint)
        steps_list = []
        scores_list = []
        score_per_step_list = []
        for _ in range(5):
            steps, score = brain.test_Network()
            steps_list.append(steps)
            scores_list.append(score)
            score_per_step_list.append(score / steps)
        
        # 计算步骤统计
        max_steps = max(steps_list)
        min_steps = min(steps_list)
        avg_steps = sum(steps_list) / 5
        play_steps_max.append(max_steps)
        play_steps_min.append(min_steps)
        play_steps_avg.append(avg_steps)
        
        # 计算得分统计
        max_score = max(scores_list)
        min_score = min(scores_list)
        avg_score = sum(scores_list) / 5
        play_scores_max.append(max_score)
        play_scores_min.append(min_score)
        play_scores_avg.append(avg_score)
        
        # 计算每步得分统计
        max_sps = max(score_per_step_list)
        min_sps = min(score_per_step_list)
        avg_sps = sum(score_per_step_list) / 5
        play_scores_per_step_max.append(max_sps)
        play_scores_per_step_min.append(min_sps)
        play_scores_per_step_avg.append(avg_sps)
        
        print(f"Checkpoint {checkpoint}:")
        print(f"  Steps: Max={max_steps}, Min={min_steps}, Avg={avg_steps:.2f}")
        print(f"  Scores: Max={max_score}, Min={min_score}, Avg={avg_score:.2f}")
        print(f"  Score/Step: Max={max_sps:.4f}, Min={min_sps:.4f}, Avg={avg_sps:.4f}")

    # 绘图
    checkpoints = [(i + 1) * 50000 for i in range(200)]
    
    plt.figure(figsize=(18, 5))
    
    # 第一个子图：步骤统计
    plt.subplot(1, 3, 1)
    plt.plot(checkpoints, play_steps_max, 'b', label='Max Steps')
    plt.plot(checkpoints, play_steps_min, 'b--', label='Min Steps')
    plt.plot(checkpoints, play_steps_avg, 'r', label='Avg Steps')
    plt.title('Steps per Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('Steps')
    plt.legend()
    
    # 第二个子图：得分统计
    plt.subplot(1, 3, 2)
    plt.plot(checkpoints, play_scores_max, 'b', label='Max Score')
    plt.plot(checkpoints, play_scores_min, 'b--', label='Min Score')
    plt.plot(checkpoints, play_scores_avg, 'r', label='Avg Score')
    plt.title('Score per Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('Score')
    plt.legend()
    
    # 第三个子图：每步得分统计
    plt.subplot(1, 3, 3)
    plt.plot(checkpoints, play_scores_per_step_max, 'b', label='Max Score/Step')
    plt.plot(checkpoints, play_scores_per_step_min, 'b--', label='Min Score/Step')
    plt.plot(checkpoints, play_scores_per_step_avg, 'r', label='Avg Score/Step')
    plt.title('Score per Step per Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('Score per Step')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./pong_performance.png')
    plt.show()
'''