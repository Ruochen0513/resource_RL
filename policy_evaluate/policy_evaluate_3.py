import numpy as np
from gymnasium.envs.toy_text.taxi import TaxiEnv
import time
import gymnasium as gym
def V_ana_evaluate(Pi, r_sa, P_ssa):
    nS, nA = Pi.shape
    P_pi = np.zeros((nS, nS))
    C_pi = np.zeros((nS, 1))
    for s in range(nS):
        P_pi[s, :] = np.sum(Pi[s].reshape(-1, 1) * P_ssa[:, s, :], axis=0)
        C_pi[s] = np.sum(Pi[s] * r_sa[s])
    V = np.dot(np.linalg.inv(np.eye(nS) - P_pi), C_pi)
    return V

def V_iter_evaluate(Pi, r_sa, P_ssa, gamma=1.0, theta=1e-6):
    nS, nA = Pi.shape
    # 构建P_pi和C_pi
    P_pi = np.zeros((nS, nS))
    C_pi = np.zeros((nS, 1))
    for s in range(nS):
        P_pi[s, :] = np.sum(Pi[s].reshape(-1, 1) * P_ssa[:, s, :], axis=0)
        C_pi[s] = np.sum(Pi[s] * r_sa[s])
    V = np.zeros((nS, 1))
    while True:
        V_new = C_pi + gamma * np.dot(P_pi, V)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    return V

def update_policy(r_sa, P_ssa, V, gamma=1.0):
    nS, nA = r_sa.shape
    Pi_new = np.zeros((nS, nA))
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            q_values[a] = r_sa[s, a] + gamma * np.dot(P_ssa[a, s, :], V.flatten())
        max_q = q_values.max()
        max_actions = np.where(q_values == max_q)[0]
        Pi_new[s, max_actions] = 1.0 / len(max_actions)
    return Pi_new

def policy_iteration(env, gamma=1, theta=1e-6, max_iter=2000):
    nS, nA = env.observation_space.n, env.action_space.n
    # 构建P_ssa和r_sa
    P_ssa = np.zeros((nA, nS, nS))
    r_sa = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for (prob, next_s, r, _) in env.P[s][a]:
                P_ssa[a, s, next_s] += prob
                r_sa[s, a] += prob * r
    # 初始化均匀策略
    Pi = np.ones((nS, nA)) / nA
    for i in range(max_iter):
        V = V_iter_evaluate(Pi, r_sa, P_ssa, gamma, theta)
        Pi_new = update_policy(r_sa, P_ssa, V, gamma)
        if np.allclose(Pi, Pi_new, atol=theta):
            print(f"策略迭代收敛于第{i+1}次迭代")
            break
        Pi = Pi_new
    return Pi, V

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
    env = TaxiEnv(render_mode='human')
    # 策略迭代
    print("策略迭代求解中...")
    optimal_pi_pi, optimal_V_pi = policy_iteration(env)
    # 策略演示
    print("\n策略演示中...")
    demonstrate_policy(env, optimal_pi_pi, render_mode='human', delay=0.5)
