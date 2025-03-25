##################迭代策略评估算法实现#########################
import numpy as np
import copy
import gymnasium as gym
import time
import seaborn as sns
#矩阵扩维和降维度
#print("P_pi",Pi[1,:],np.expand_dims(Pi[1,:],axis=0))
#print(np.dot(np.expand_dims(Pi[1,:],axis=0),P_ssa[:,1,:]).squeeze())
class FrozenLakeEnv:
    def __init__(self):
        # 创建 FrozenLake 环境
        self.env = gym.make('FrozenLake-v1', is_slippery=True)
        
        # 初始化行为值函数
        self.qvalue = -0.01 * np.zeros((16, 4))
        # 初始化每个状态-动作对的次数
        self.n = 0 * np.ones((16, 4))
        
        self.states = np.arange(16)
        self.actions = np.arange(4)
        self.gamma = 0.99
        
        # 初始化策略
        self.Pi = 0.25 * np.ones((16, 4))
        self.Greedy_Pi = np.zeros((16, 4))
        self.cur_state = 0
        self.old_policy = np.ones((16, 4))
        self.epsilon = 0.5

    def reset(self):
        # 重置环境和行为值函数
        self.qvalue = -0.01 * np.zeros((16, 4))
        self.n = 0 * np.ones((16, 4))
        self.cur_state, _ = self.env.reset()
        return self.cur_state

    def sample_action(self, state):
        # 根据当前策略选择动作
        action = np.random.choice(self.actions, p=self.Pi[state,:])
        return action

    def step(self, action):
        # 执行动作并获取下一个状态和奖励
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def update_policy(self):
        for i in range(16):
            self.Pi[i,:] = 0
            max_num = np.argmax(self.qvalue[i,:])
            self.Pi[i, max_num] = 1

    def get_greedy_policy(self):
        for i in range(16):
            self.Greedy_Pi[i,:] = 0
            max_num = np.argmax(self.qvalue[i,:])
            self.Greedy_Pi[i, max_num] = 1
        return self.Greedy_Pi

    def update_epsilon_greedy(self):
        for i in range(16):
            self.Pi[i,:] = self.epsilon/4
            max_num = np.argmax(self.qvalue[i,:])
            self.Pi[i, max_num] = self.epsilon/4 + (1-self.epsilon)

    def MC_learning(self):
        num = 0
        while num < 6000:
            num += 1
            flag = False
            state_traj = []
            action_traj = []
            reward_traj = []
            g = 0
            episode_num = 0
            
            # 重置环境
            self.cur_state, _ = self.env.reset()
            
            while not flag and episode_num < 200:
                cur_action = self.sample_action(self.cur_state)
                state_traj.append(self.cur_state)
                action_traj.append(cur_action)
                
                next_state, reward, flag = self.step(cur_action)
                reward_traj.append(reward)
                self.cur_state = next_state
                episode_num += 1

            # 更新行为值函数
            for i in reversed(range(len(state_traj))):
                self.n[state_traj[i], action_traj[i]] += 1.0
                g *= self.gamma
                g += reward_traj[i]
                self.qvalue[state_traj[i], action_traj[i]] = \
                    (self.qvalue[state_traj[i], action_traj[i]] * (self.n[state_traj[i], action_traj[i]]-1) + g) / \
                    self.n[state_traj[i], action_traj[i]]

            if num % 2001 == 0:
                self.old_policy = copy.deepcopy(self.Pi)
                self.update_epsilon_greedy()
                self.epsilon = self.epsilon * 0.90
                self.n = np.zeros((16, 4))

def q_ana_evaluate(Pi,r_sa,P_ssa):
    P_pi = np.zeros((16, 16))
    C_pi = np.zeros((16, 1))
    for i in range(16):
        # 计算pi(a|s)*p(s'|s,a)
        P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
        # 计算pi(a|s)*r(s,a)
        C_pi[i, :] = np.dot(r_sa[i, :], Pi[i, :])
    ############解析法计算值函数######################
    M = np.eye(16) - P_pi
    I_M = np.linalg.inv(M)
    V = np.dot(I_M, C_pi)
    #计算行为值函数
    q_value = np.zeros((16, 4))
    for i in range(16):
        q_sa = np.zeros((1, 4))
        for j in range(4):
            Pi[i, :] = 0
            Pi[i, j] = 1
            P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
            vi = np.dot(r_sa[i, :], Pi[i, :]) + np.dot(P_pi[i, :], V.squeeze())
            q_sa[0, j] = vi
        q_value[i, :] = q_sa[0, :]
    return q_value

def visualize_policy(policy, env):
    # 获取环境布局描述
    desc = env.unwrapped.desc  # 4x4的字符矩阵
    
    # 动作到箭头的映射
    arrow_map = {
        0: '←',  # 左
        1: '↓',  # 下
        2: '→',  # 右
        3: '↑'   # 上
    }
    
    # 构建可视化矩阵
    grid = []
    for row in range(4):
        current_row = []
        for col in range(4):
            state = row * 4 + col
            cell_char = desc[row, col].decode('utf-8')
            
            # 处理特殊单元格
            if cell_char == 'H':
                current_row.append('⛳')  # 冰洞
            elif cell_char == 'G':
                current_row.append('🎯')  # 目标
            else:
                # 获取该状态的最优动作
                action = np.argmax(policy[state])
                current_row.append(arrow_map[action])
        grid.append(current_row)
        # 打印可视化结果
    print("\n最优策略可视化：")
    print("+" + "-"*13 + "+")
    for i, row in enumerate(grid):
        print("| " + " | ".join(row) + " |")
        if i < 3: 
            print("|" + "----+---+---+----" + "|")
    print("+" + "-"*13 + "+")

def visualize_policy_matrix(policy, env):
    import matplotlib.pyplot as plt
    import numpy as np
    
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
            action = np.argmax(policy[state])
            plt.text(j, 4-i, arrow_symbols[action], 
                    ha='center', va='center', fontsize=20)
    
    plt.title('Policy Visualization')
    plt.tight_layout()
    plt.show()

def test_policy(policy, num_episodes=1000):
    # 创建无渲染的测试环境（参数与训练环境保持一致）
    test_env = gym.make('FrozenLake-v1', 
                      is_slippery=True,
                      render_mode=None).unwrapped
    success_count = 0
    
    for _ in range(num_episodes):
        state, _ = test_env.reset()
        done = False
        
        while not done:
            action = np.argmax(policy[state])
            next_state, reward, done, truncated, _ = test_env.step(action)
            state = next_state
            
            # 判断是否成功到达目标
            if done and reward == 1:
                success_count += 1
                break
    test_env.close()
    return success_count / num_episodes
    

def plot_heatmaps(qvalue, visit_counts):
    import matplotlib.pyplot as plt
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Q值热力图
    sns.heatmap(np.max(qvalue, axis=1).reshape(4, 4), annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("State Value Heatmap (Max Q)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # 访问频次热力图
    sns.heatmap(np.sum(visit_counts, axis=1).reshape(4, 4), annot=True, fmt=".0f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Visit Frequency Heatmap")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
        
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    env = FrozenLakeEnv()
    env.reset()
    print("初始策略:", env.Pi)
    start = time.time()
    env.MC_learning()
    end = time.time()
    print("最终策略:\n", np.around(env.Pi, 2))
    print("当前Q值: \n", np.around(env.qvalue, 2))
    print("访问频次:", env.n)
    print("探索因子：", env.epsilon)
    print("贪婪策略:", env.get_greedy_policy())
    print("运行时间：", end-start, "s")
    # 测试成功率代码
    success_rate = test_policy(env.get_greedy_policy(), num_episodes=1000)
    print(f"\n最优策略成功率统计: {success_rate*100:.2f}% (1000次尝试)")
    # 添加可视化
    env2 = gym.make('FrozenLake-v1', is_slippery=True)
    visualize_policy_matrix(env.get_greedy_policy(), env2)
    plot_heatmaps(env.qvalue, env.n)