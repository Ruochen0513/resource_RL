import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))
        std = self.log_std.exp()
        return mean, std

# 训练函数
def train():
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 超参数
    gamma = 0.99
    learning_rate = 3e-4
    episodes = 4000
    save_interval = 100

    # 初始化策略网络并转移到GPU
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # 记录奖励和损失
    rewards_list = []
    loss_list = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        log_probs = []
        rewards_episode = []

        while not done:
            # 状态转移到GPU
            state_tensor = torch.FloatTensor(state).to(device)
            mean, std = policy(state_tensor)
            
            # 采样动作
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            
            # 动作转移到CPU执行
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards_episode.append(reward)
            state = next_state

        # 计算折扣回报
        discounted_rewards = []
        R = 0
        for r in reversed(rewards_episode):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        # 数据转移到GPU并标准化
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # 计算损失
        policy_loss = torch.stack([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]).sum()
        
        # 记录指标
        total_reward = sum(rewards_episode)
        rewards_list.append(total_reward)
        loss_list.append(policy_loss.item())

        # 反向传播
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # 保存模型
        if (episode + 1) % save_interval == 0:
            torch.save(policy.state_dict(), f'./models/model_{episode+1}.pth')
            print(f"模型已保存至 _model_{episode+1}.pth")

        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Loss: {policy_loss.item():.2f}")

    env.close()

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_list)
    plt.xlabel('Episode')
    plt.ylabel('Policy Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_curve.png')
    plt.show()

if __name__ == '__main__':
    train()