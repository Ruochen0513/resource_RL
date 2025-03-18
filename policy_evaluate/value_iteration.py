################## 值迭代算法实现（含gamma）#########################
import numpy as np
import gymnasium as gym

def update_policy(r_sa, P_ssa, V, gamma):  # 添加gamma参数
    Pi_new = np.zeros((16, 4))
    for i in range(16):
        q_sa = np.zeros(4)
        for j in range(4):
            # 直接计算每个动作的Q值
            P_pi = P_ssa[j, i, :]  # 动作j的转移概率
            q_sa[j] = r_sa[i, j] + gamma * np.dot(P_pi, V.squeeze())  # 添加gamma
        max_num = np.argmax(q_sa)
        Pi_new[i, max_num] = 1
    return Pi_new

def value_iteration(Pi_0, r_sa, P_ssa, V_init, gamma, tol=1e-6):  # 添加gamma参数
    V_cur = V_init
    iter_num = 0
    
    while True:
        # 值函数更新
        V_next = np.zeros_like(V_cur)
        for i in range(16):
            q_values = [r_sa[i, a] + gamma * np.dot(P_ssa[a, i, :], V_cur.squeeze()) 
                       for a in range(4)]
            V_next[i] = np.max(q_values)  # 直接取最大Q值
        
        delta = np.linalg.norm(V_next - V_cur)
        V_cur = V_next
        iter_num += 1
        
        if delta < tol:
            break
    
    # 最终策略提取
    Pi_optim = update_policy(r_sa, P_ssa, V_cur, gamma)
    return Pi_optim, iter_num

# 成功率测试函数
def test_policy(policy, num_episodes=1000):
    test_env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    success = 0
    
    for _ in range(num_episodes):
        state, _ = test_env.reset()
        done = False
        
        while not done:
            action = np.argmax(policy[state])
            next_state, reward, done, _, _ = test_env.step(action)
            state = next_state
            
            if done and reward == 1:
                success += 1
                
    test_env.close()
    return success / num_episodes

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

def show(env, policy, render=False):
    state, _ = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        action = np.argmax(policy[state])
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        if truncated:
            break

if __name__ == '__main__':
    # 创建环境
    env = gym.make('FrozenLake-v1', map_name="4x4", 
                  is_slippery=True, render_mode="human").unwrapped
    
    # 初始化转移矩阵和奖励矩阵
    nS = env.observation_space.n
    nA = env.action_space.n
    P_ssa = np.zeros((nA, nS, nS))
    r_sa = np.zeros((nS, nA))
    
    for s in range(nS):
        for a in range(nA):
            for (prob, next_s, reward, _) in env.P[s][a]:
                P_ssa[a, s, next_s] += prob
                r_sa[s, a] += prob * reward

    # 执行值迭代
    Pi_optim, iter_num = value_iteration(
        Pi_0=np.ones((nS, nA))/nA,  # 初始随机策略
        r_sa=r_sa,
        P_ssa=P_ssa,
        V_init=np.zeros((nS, 1)),
        gamma=1  # 可调整参数
    )
    
    # 输出结果
    print(f"迭代次数: {iter_num}")
    print("最优策略矩阵:")
    print(Pi_optim)
    
    # 测试成功率
    success_rate = test_policy(Pi_optim)
    print(f"\n成功率: {success_rate*100:.2f}%")

    # 可视化策略
    visualize_policy(Pi_optim, env)
    # 策略演示
    show(env, Pi_optim)
    

