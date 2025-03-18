# TODO:使用策略迭代和值迭代算法求解上次挑选的Gym环境的最优策略

##################迭代策略评估算法实现#########################
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
#矩阵扩维和降维度
#print("P_pi",Pi[1,:],np.expand_dims(Pi[1,:],axis=0))
#print(np.dot(np.expand_dims(Pi[1,:],axis=0),P_ssa[:,1,:]).squeeze())
##########利用解析法进行策略评估###########
def V_ana_evaluate(Pi,r_sa,P_ssa,gamma):
    P_pi = np.zeros((16, 16))
    C_pi = np.zeros((16, 1))
    for i in range(16):
        # 计算pi(a|s)*p(s'|s,a)
        P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
        # 计算pi(a|s)*r(s,a)
        C_pi[i, :] = np.dot(r_sa[i, :], Pi[i, :])
    ############解析法计算值函数######################
    M = np.eye(16) - P_pi *gamma
    I_M = np.linalg.inv(M)
    V = np.dot(I_M, C_pi)
    return V
##########利用数值迭代法进行策略评估###########
def V_iter_evaluate(Pi,r_sa,P_ssa,gamma,V_init=np.zeros((16,1))):
    # 初始化当前值函数
    V_cur = V_init
    #计算C_pi和P_pi
    P_pi = np.zeros((16, 16))
    C_pi = np.zeros((16, 1))
    for i in range(16):
        # 计算pi(a|s)*p(s'|s,a)
        P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
        # 计算pi(a|s)*r(s,a)
        C_pi[i, :] = np.dot(r_sa[i, :], Pi[i, :])
    V_next = C_pi + np.dot(P_pi, V_cur)*gamma
    # 计算迭代一次的误差
    delta = np.linalg.norm(V_next - V_cur)
    num=0
    while delta > 1e-6:
        print("num=",num)
        print("V",V_cur)
        V_cur = V_next
        V_next = C_pi + np.dot(P_pi, V_cur) *gamma
        delta = np.linalg.norm(V_next - V_cur)
        num+=1
    print("num:",num)
    print("V_cur",V_cur)
    return V_cur
#############策略改进源代码##########         贪婪策略
def update_policy(r_sa,P_ssa,V,gamma):
    Pi_new = np.zeros((16, 4))
    Pi = np.zeros((16,4))
    # 计算C_pi和P_pi
    P_pi = np.zeros((16, 16))
    for i in range(16):
        q_sa = np.zeros((1, 4))
        for j in range(4):
            Pi[i, :] = 0
            Pi[i, j] = 1
            P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
            vi = np.dot(r_sa[i, :], Pi[i, :]) + np.dot(P_pi[i, :], V.squeeze())*gamma
            q_sa[0, j] = vi
        max_num = np.argmax(q_sa)
        Pi_new[i, max_num] = 1
    return Pi_new
###########策略迭代算法##########
def policy_iteration(Pi,r_sa,P_ssa,gamma):
    Pi_cur = Pi
    #策略评估
    V_cur = V_iter_evaluate(Pi_cur,r_sa,P_ssa,gamma)
    #V_cur = V_ana_evaluate(Pi_cur, r_sa, P_ssa,gamma)
    #策略改进
    Pi_new = update_policy(r_sa,P_ssa,V_cur,gamma)
    delta =  np.linalg.norm(Pi_new-Pi_cur)
    iter_num = 1
    while delta>1e-6:
        Pi_cur = Pi_new
        # 策略评估
        V_cur = V_iter_evaluate(Pi_cur, r_sa, P_ssa,gamma,V_cur)
        #V_cur = V_ana_evaluate(Pi_cur, r_sa, P_ssa)
        # 策略改进
        Pi_new = update_policy(r_sa, P_ssa, V_cur,gamma)
        delta = np.linalg.norm(Pi_new - Pi_cur)
        iter_num=iter_num+1
    return Pi_cur,iter_num

def show(env, policy, render=False):
    state, _ = env.reset()
    done = False
    while True:
        if render:
            env.render()
        action = np.argmax(policy[state])
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        if truncated:
            break

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

if __name__ == '__main__':
    # 创建环境
    env = gym.make('FrozenLake-v1',is_slippery=True,render_mode = "human").unwrapped
    nS = env.observation_space.n  # 状态数 (16)
    nA = env.action_space.n       # 动作数 (4)
    gamma = 1
    # 初始化转移矩阵和奖励矩阵
    P_ssa = np.zeros((nA, nS, nS))
    r_sa = np.zeros((nS, nA))
    Pi = np.ones((nS, nA)) / nA
    # 填充P_ssa和r_sa
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for (prob, next_s, reward, done) in transitions:
                P_ssa[a, s, next_s] += prob
                r_sa[s, a] += prob * reward  # 计算期望奖励
    Pi_optim,iter_num = policy_iteration(Pi,r_sa,P_ssa,gamma)
    print("最优策略矩阵为：",Pi_optim,"迭代轮数为：",iter_num)
    
    # 测试成功率代码
    success_rate = test_policy(Pi_optim, num_episodes=1000)
    print(f"\n最优策略成功率统计: {success_rate*100:.2f}% (1000次尝试)")
    # 在终端可视化最优策略
    print("\n可视化最优策略：")
    visualize_policy(Pi_optim, env)
    # 显示最优策略
    show(env,Pi_optim)




