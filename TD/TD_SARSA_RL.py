import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
class FrozenLake:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes):
        self.env = env
        self.state_n = self.env.observation_space.n         # 状态空间数量
        self.action_n = self.env.action_space.n             # 动作空间数量
        self.alpha = alpha                                    # 学习率
        self.epsilon = epsilon                                # 探索率
        self.gamma = gamma                                    # 折扣因子
        self.num_episodes = num_episodes                      # 训练轮数
        self.qvalue = np.zeros((self.state_n, self.action_n))      # 初始化Q值函数
    ###根据Q值获取贪婪策略
    def get_greedy_policy(self):
        greedy_policy = np.zeros(self.state_n)
        for i in range(self.state_n):
            greedy_policy[i] = np.argmax(self.qvalue[i])
        return greedy_policy
    ###根据epsilon_greedy策略获取动作
    def epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_n)
        else:
            max_actions = np.where(self.qvalue[state] == np.max(self.qvalue[state]))[0]
            return np.random.choice(max_actions)
    ### SARSA算法
    def Sarsa(self):
        for episode in tqdm(range(self.num_episodes)):
            state = self.env.reset()[0]
            action = self.epsilon_greedy_action(state)
            done = False            ## 终止标志
            while not done:
                # 交互一步
                next_state, reward, done, _, _ = self.env.step(action)
                # 选择下一个动作
                next_action = self.epsilon_greedy_action(next_state)
                # 更新Q值
                td_target = reward + self.gamma * self.qvalue[next_state, next_action]
                td_error = td_target - self.qvalue[state, action]
                self.qvalue[state, action] += self.alpha * td_error
                # 更新状态和动作
                state = next_state
                action = next_action
    ### 测试指定策略的成功率
    def test_policy(self, policy, num_tests=1000):
        success_count = 0
        for _ in range(num_tests):
            state = self.env.reset()[0]
            done = False
            while not done:
                action = policy[state]
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
            if reward == 1:  # 成功到达终点
                success_count += 1
        return success_count / num_tests
def plot_q_heatmaps(q_table):
    action_names = ['Left', 'Down', 'Right', 'Up']
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.3)  # 为颜色条腾出空间
    
    # 获取最大最小值统一色标
    vmin = np.min(q_table)
    vmax = np.max(q_table)
    
    # 绘制每个动作的热力图
    for action in range(4):
        q_grid = q_table[:, action].reshape(4, 4)
        im = axes[action].imshow(q_grid, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[action].set_title(action_names[action], pad=15)
        axes[action].set_xticks(np.arange(4))
        axes[action].set_yticks(np.arange(4))
        
        # 添加坐标轴标签（只在边缘显示）
        if action == 0:
            axes[action].set_ylabel('Row', labelpad=10)
        axes[action].set_xlabel('Column', labelpad=10)

        # 添加数值标注
        for i in range(4):
            for j in range(4):
                axes[action].text(j, i, f"{q_grid[i, j]:.2f}",
                                ha="center", va="center", 
                                color="w", fontsize=8)

    # 添加横向颜色条
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Q-value Scale', labelpad=10)
    
    plt.suptitle("Q-value Heatmaps with Horizontal Colorbar", y=0.95)
    plt.show()
def visualize_policy_matrix(policy, env):
    # 获取环境布局
    desc = env.unwrapped.desc
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))
    # 设置网格
    ax.grid(True)
    ax.set_xticks(np.arange(-0.5, 4.5, 1))
    ax.set_yticks(np.arange(-0.5, 4.5, 1))
    
    # 隐藏刻度标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 动作到箭头的映射
    arrow_symbols = {
        0: '←',  # 左
        1: '↓',  # 下
        2: '→',  # 右
        3: '↑'   # 上
    }
    
    # 填充颜色和箭头
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            cell = desc[i][j].decode('utf-8')
            
            # 设置单元格颜色
            if cell == 'H':
                color = '#ff6666'  # 深红色表示洞
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [3.5-i, 3.5-i, 4.5-i, 4.5-i], 
                        color=color)
            elif cell == 'G':
                color = '#ccffcc'  # 浅绿色表示目标
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [3.5-i, 3.5-i, 4.5-i, 4.5-i], 
                        color=color)
            else:
                color = '#ffffff'  # 白色表示可移动格子
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [3.5-i, 3.5-i, 4.5-i, 4.5-i], 
                        color=color)
            
            # 添加箭头
            action = policy[state]
            plt.text(j, 4-i, arrow_symbols[action], 
                    ha='center', va='center', fontsize=20)
    
    plt.title('Policy Visualization')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    num_episodes = 40000
    # 创建SARSA实例并训练
    sarsa_agent = FrozenLake(env, alpha, gamma, epsilon, num_episodes)
    start_time = time.time()
    sarsa_agent.Sarsa()
    end_time = time.time()
    # 输出Q值函数
    print("Q-value Function:")
    print(sarsa_agent.qvalue)
    # 获取贪婪策略
    greedy_policy = sarsa_agent.get_greedy_policy()
    print("Greedy Policy:", greedy_policy)
    # 输出训练时间
    print(f"训练时间: {end_time - start_time:.2f}秒")
    # 测试策略
    success_rate = sarsa_agent.test_policy(greedy_policy)
    print(f"依照当前策略测试1000次的成功率： {success_rate * 100:.2f}%")
    # 绘制Q值热力图
    plot_q_heatmaps(sarsa_agent.qvalue)
    # 可视化最优策略
    visualize_policy_matrix(greedy_policy, env)