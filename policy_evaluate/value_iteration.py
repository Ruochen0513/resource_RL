import numpy as np
from gymnasium.envs.toy_text.taxi import TaxiEnv
import time
def update_policy(r_sa, P_ssa, V, gamma=1):
    num_states, num_actions = r_sa.shape
    Pi_new = np.zeros((num_states, num_actions))
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            q_values[a] = r_sa[s, a] + gamma * np.dot(P_ssa[a, s, :], V)
        best_action = np.argmax(q_values)
        Pi_new[s, best_action] = 1.0
    return Pi_new

def value_iteration(Pi_0, r_sa, P_ssa, V_init, gamma=1, tol=1e-6, max_iter=10000):
    Pi_cur = Pi_0.copy()
    V_cur = V_init.copy()
    iter_num = 0

    while True:
        # 策略评估
        C_pi = (Pi_cur * r_sa).sum(axis=1, keepdims=True)
        P_pi = np.einsum('sa,ast->st', Pi_cur, P_ssa)
        V_hat = C_pi + gamma * np.dot(P_pi, V_cur)
        
        # 检查收敛
        delta = np.max(np.abs(V_hat - V_cur))
        V_cur = V_hat.copy()
        
        # 策略改进
        Pi_new = update_policy(r_sa, P_ssa, V_cur.squeeze(), gamma)
        
        iter_num += 1
        if delta < tol or iter_num >= max_iter:
            break
            
        Pi_cur = Pi_new.copy()

    return Pi_cur, iter_num


################## 策略演示函数 #########################
def demonstrate_policy(env, policy, render_mode='human', delay=0.5):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        # 获取当前状态的有效动作
        valid_actions = np.where(env.action_mask(state))[0]
        
        # 选择最优策略中的有效动作
        action_probs = policy[state]
        valid_probs = action_probs[valid_actions]
        action = valid_actions[np.argmax(valid_probs)]
        
        next_state, reward, done, _, _ = env.step(action)
        
        if render_mode == 'human':
            env.render()
            time.sleep(delay)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            print(f"演示完成! 总奖励: {total_reward}, 步数: {steps}")
            break

if __name__ == '__main__':
    # 初始化环境
    env = TaxiEnv(render_mode='human')
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # 构建状态转移矩阵和奖励矩阵
    P_ssa = np.zeros((num_actions, num_states, num_states))
    r_sa = np.zeros((num_states, num_actions))
    
    for s in range(num_states):
        for a in range(num_actions):
            transitions = env.P[s][a]
            for prob, next_state, reward, _ in transitions:
                P_ssa[a, s, next_state] += prob
                r_sa[s, a] += prob * reward

    # 初始化参数
    Pi_0 = np.ones((num_states, num_actions)) / num_actions  
    # 均匀随机策略
    V_init = np.zeros((num_states, 1))
    
    # 运行值迭代
    Pi_optim, iter_num = value_iteration(Pi_0, r_sa, P_ssa, V_init, gamma=1)
    
    # 输出结果
    print(f"收敛所需迭代次数: {iter_num}")
    demonstrate_policy(env, Pi_optim, render_mode='human', delay=0.5)