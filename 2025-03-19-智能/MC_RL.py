##################迭代策略评估算法实现#########################
import numpy as np
import copy
import gymnasium as gym
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
        self.cur_state = 0
        self.old_policy = np.ones((16, 4))
        
    def reset(self):
        # 重置环境和行为值函数
        self.qvalue = -0.01 * np.zeros((16, 4))
        self.n = 0 * np.ones((16, 4))
        self.cur_state, _ = self.env.reset()
        return self.cur_state
        
    def explore_init(self):
        # 随机选择初始状态和动作
        self.cur_state, _ = self.env.reset()
        a0 = np.random.choice(self.actions, p=[0.25, 0.25, 0.25, 0.25])
        return self.cur_state, a0
        
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
        # 贪婪策略更新
        for i in range(16):
            self.Pi[i,:] = 0
            max_num = np.argmax(self.qvalue[i,:])
            self.Pi[i, max_num] = 1
            
    def update_epsilon_greedy(self):
        # ε-贪婪策略更新
        epsilon = 0.2
        for i in range(16):
            self.Pi[i,:] = epsilon/4
            max_num = np.argmax(self.qvalue[i,:])
            self.Pi[i, max_num] = 0.85

    def MC_learning(self):
        num = 0
        while num < 6000:
            num += 1
            flag = False
            # 采样一条轨迹
            state_traj = []
            action_traj = []
            reward_traj = []
            g = 0
            episode_num = 0
            
            # 探索初始化
            cur_state, cur_action = self.explore_init()
            
            while not flag and episode_num < 200:
                if episode_num > 0:
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

            # 定期更新策略
            if num % 501 == 0:
                self.old_policy = copy.deepcopy(self.Pi)
                self.update_policy()
                self.n = np.zeros((16, 4))
                delta = np.linalg.norm(self.old_policy - self.Pi)
                print("delta", delta)

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
                color = '#ffcccc'  # 浅红色表示洞
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
            if cell != 'H' and cell != 'G':
                action = np.argmax(policy[state])
                plt.text(j, 4-i-0.5, arrow_symbols[action], 
                        ha='center', va='center', fontsize=20)
    
    plt.title('最优策略可视化')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 实例化对象
    env = FrozenLakeEnv()
    env.reset()
    print("初始策略:", env.Pi)
    env.MC_learning()
    print("最终策略:", env.Pi)
    print("最优Q值:", env.qvalue)
    print("访问频次:", env.n)

    # 添加可视化
    env2 = gym.make('FrozenLake-v1', is_slippery=True)
    visualize_policy_matrix(env.Pi, env2)