import os
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
RENDER = False
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import os
loss_history = []
#初始化神经网络层权重
def layer_init(layer,std = np.sqrt(2),bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight,std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer
class Experience_Buffer():
    def __init__(self, buffer_size=40000):
      self.buffer = []
      self.buffer_size = buffer_size
      # 定义状态和动作维度
      self.obs_dim = 8  # LunarLanderContinuous-v2 状态维度
      self.act_dim = 2  # LunarLanderContinuous-v2 动作维度
    
    def add_experience(self, state, action, reward, next_state, done):
      """添加一条经验到缓冲区，包含终止信号done"""
      experience = {
          'state': state,
          'action': action,
          'reward': reward,
          'next_state': next_state,
          'done': done  # 新增终止信号
      }
      
      # 移除旧样本以保持缓冲区大小
      if len(self.buffer) >= self.buffer_size:
          self.buffer.pop(0)
          
      self.buffer.append(experience)
    
    def sample(self, batch_size):
      """从缓冲区中采样一批经验"""
      # 确保缓冲区有足够数据
      batch_size = min(batch_size, len(self.buffer))
      
      # 随机采样索引
      indices = np.random.choice(len(self.buffer), batch_size, replace=False)
      
      # 准备批次数据
      states = np.zeros((batch_size, self.obs_dim))
      actions = np.zeros((batch_size, self.act_dim))
      rewards = np.zeros((batch_size, 1))
      next_states = np.zeros((batch_size, self.obs_dim))
      dones = np.zeros((batch_size, 1))  # 新增终止信号数组
      
      # 填充数据
      for i, idx in enumerate(indices):
          experience = self.buffer[idx]
          states[i] = experience['state']
          actions[i] = experience['action']
          rewards[i] = experience['reward']
          next_states[i] = experience['next_state']
          dones[i] = float(experience['done'])  # 转换为float类型
      
      # 转换为PyTorch张量
      states = torch.as_tensor(states, dtype=torch.float32)
      actions = torch.as_tensor(actions, dtype=torch.float32)
      rewards = torch.as_tensor(rewards, dtype=torch.float32)
      next_states = torch.as_tensor(next_states, dtype=torch.float32)
      dones = torch.as_tensor(dones, dtype=torch.float32)  # 转换为张量
      
      return states, actions, rewards, next_states, dones
#策略网络类，构建策略，并进行采样
class Actor_Net(nn.Module):
  def __init__(self,obs_dim,act_dim,hidden_sizes,action_bounds=None):
    super(Actor_Net,self).__init__()
    self.act_dim = act_dim
    self.action_bounds = action_bounds
    #均值网络，利用前向神经网络
    self.actor_net = nn.Sequential(
      layer_init(nn.Linear(obs_dim,hidden_sizes[0])),
      nn.ReLU(),
      layer_init(nn.Linear(hidden_sizes[0],hidden_sizes[1])),
      nn.ReLU(),
      layer_init(nn.Linear(hidden_sizes[1],act_dim),std=0.01),
      nn.Tanh()
    )
    self.actor_net.requires_grad_()

  def forward(self, obs):
    # 输出动作，并根据动作边界进行缩放
    action_raw = self.actor_net(obs)
    if self.action_bounds is not None:
      action_low, action_high = self.action_bounds
      # 确保动作边界不包含梯度
      action_low = action_low.detach() if isinstance(action_low, torch.Tensor) else torch.tensor(action_low, dtype=torch.float32)
      action_high = action_high.detach() if isinstance(action_high, torch.Tensor) else torch.tensor(action_high, dtype=torch.float32)
      # 应用动作范围
      action = action_raw * 0.5 * (action_high - action_low) + 0.5 * (action_high + action_low)
    else:
      action = action_raw
    return action
    
  #采样一个动作，不计算梯度,产生数据样本时使用
  def get_a(self,obs):
    with torch.no_grad():
      action = self.forward(obs)
    return action
#值函数网络类，构建值函数
class Critic_Net(nn.Module):
  def __init__(self,obs_dim,action_dim,hidden_sizes):
    super(Critic_Net,self).__init__()
    self.layer_1 = layer_init(nn.Linear(obs_dim,hidden_sizes[0]))
    self.layer_2 = layer_init(nn.Linear(action_dim,hidden_sizes[0]))
    self.layer_3 = layer_init(nn.Linear(hidden_sizes[0],hidden_sizes[1]))
    self.output = layer_init(nn.Linear(hidden_sizes[1],1))
  #返回值函数预测值，计算梯度，计算值函数损失时使用
  def forward(self,obs,action):
    q =self.layer_1(obs)+self.layer_2(action)
    q = torch.relu(self.layer_3(q))
    q =self.output(q)
    return q
  #计算值函数的数值，不计算梯度，计算目标函数时使用
  def get_q(self,obs,action):
    with torch.no_grad():
      q = self.forward(obs,action)
    return q.numpy()
#PPO算法类，实现策略的更新
class DDPG():
  def __init__(self, env):
    self.env = env
    self.gamma = 0.99
    self.exp_buffer = Experience_Buffer()
    self.obs_dim = env.observation_space.shape[0]  # 获取观察空间维度
    self.act_dim = env.action_space.shape[0]  # 获取动作空间维度
    
    # 获取动作范围
    self.action_bounds = [
        env.action_space.low,
        env.action_space.high
    ]
    
    # 调整网络大小
    self.hidden = [64, 64]  # 扩大网络容量
    self.pi_lr = 3e-4  # 按PPO的学习率设置
    self.critic_lr = 1e-3
    self.tau = 0.005 
    self.batch_num = 64
    self.epochs = 4000  # 设置更多训练轮次
    self.save_freq = 10
    
    # 训练监控
    self.return_traj = []
    self.recent_rewards = []  # 记录最近的奖励
    self.success_threshold = 200  # LunarLander的成功阈值
    
    # 初始化网络
    self.actor = Actor_Net(self.obs_dim, self.act_dim, self.hidden, self.action_bounds)
    self.actor_target = Actor_Net(self.obs_dim, self.act_dim, self.hidden, self.action_bounds)
    self.pi_optimizer = Adam(self.actor.parameters(), lr=self.pi_lr)
    
    self.critic = Critic_Net(self.obs_dim, self.act_dim, self.hidden)
    self.critic_target = Critic_Net(self.obs_dim, self.act_dim, self.hidden)
    self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
    
    # 硬拷贝参数到目标网络
    self.hard_update(self.actor_target, self.actor)
    self.hard_update(self.critic_target, self.critic)
    
    # 设置模型保存路径
    self.training_path = 'D:\\code\\resource_RL\\ddpg\\models'
    os.makedirs(self.training_path, exist_ok=True)
    self.actor_filename = 'ddpg_lunar_actor.pth'  # 更改文件名
    self.critic_filename = 'ddpg_lunar_critic.pth'
    
    # 跟踪最佳模型
    self.best_return = -float('inf')
    self.best_model_path = None
    
  # 计算策略损失函数
  def compute_loss_pi(self, obs):
    act = self.actor(obs)
    q = self.critic(obs,act)
    loss_pi = -torch.mean(q)
    return loss_pi
    
  # 计算值损失
  def compute_loss_critic(self, obs, act, reward, obs_next, done):
    # 考虑终止状态
    a_next = self.actor_target(obs_next)
    q_target = self.critic_target(obs_next, a_next).detach()
    # 如果是终止状态，下一状态没有回报
    backup = reward + self.gamma * (1 - done) * q_target
    return ((self.critic(obs, act) - backup) ** 2).mean()
    
  # 采集数据并更新
  def update(self):
    # 从经验回放中采样数据
    batch_state, batch_act, batch_reward, batch_state_next, batch_done = self.exp_buffer.sample(self.batch_num)
      
    # 优化评价网络 - 先更新critic，使用更好的critic计算policy损失
    loss_critic = self.compute_loss_critic(batch_state, batch_act, batch_reward, batch_state_next, batch_done)
    self.critic_optimizer.zero_grad()
    loss_critic.backward()
    # 限制梯度，提高稳定性
    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
    self.critic_optimizer.step()
    
    # 优化策略参数
    loss_pi = self.compute_loss_pi(batch_state)
    self.pi_optimizer.zero_grad()
    loss_pi.backward()
    # 限制梯度，提高稳定性
    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
    self.pi_optimizer.step()
    
    return loss_pi.item(), loss_critic.item()
    
  # 软更新
  def soft_update(self, target_net, eva_net, tau):
    for target_param, param in zip(target_net.parameters(), eva_net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
      
  # 硬更新
  def hard_update(self, target_net, eva_net):
    for target_param, param in zip(target_net.parameters(), eva_net.parameters()):
      target_param.data.copy_(param.data)
      
  # ddpg训练算法
  def ddpg_train(self, epochs=None):
    if epochs:
      self.epochs = epochs
      
    print("开始LunarLanderContinuous-v2 DDPG训练...")
    print(f"目标：回报达到 {self.success_threshold}")
      
    # 记录训练损失
    pi_losses = []
    critic_losses = []
    
    # 设置Ornstein-Uhlenbeck噪声进程（更适合连续控制任务）
    class OUNoise:
      def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
          
      def reset(self):
        self.state = np.copy(self.mu)
          
      def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
    # 初始化噪声对象
    noise = OUNoise(self.act_dim)
    
    # 初始化学习进度控制
    update_counter = 0
    target_update_freq = 10 
    
    for epoch in range(self.epochs):
      observation, _ = self.env.reset()
      noise.reset()  # 重置噪声
      episode_return = 0
      done = False
      steps = 0
      
      # 动态噪声衰减
      noise_scale = max(0.005, 0.3 * np.exp(-epoch / 200))
      
      # 记录每轮的损失
      epoch_pi_losses = []
      epoch_critic_losses = []
      
      while not done:
        # 获取动作
        state = observation.reshape(1, -1)
        action = self.actor(torch.as_tensor(state, dtype=torch.float32))
        action = action.detach().numpy()
        
        # 应用OU噪声代替简单的高斯噪声
        action = action + noise_scale * noise.sample()
        
        # 裁剪动作到合法范围
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        
        # 环境交互
        observation_next, reward, terminated, truncated, _ = self.env.step(action[0])
        done = terminated or truncated
        # 累积回报
        episode_return += reward
        steps += 1
        
        # 存储经验
        self.exp_buffer.add_experience(observation, action[0], reward, observation_next, done)
        
        # 当经验缓冲区足够大时开始更新
        if len(self.exp_buffer.buffer) > self.batch_num * 5:  # 确保有足够数据但不要等待太长
          # 多次更新，提高样本效率
          for _ in range(1):  # 每收集1个样本更新1次网络
            pi_loss, critic_loss = self.update()
            epoch_pi_losses.append(pi_loss)
            epoch_critic_losses.append(critic_loss)
            update_counter += 1
            
            # 周期性更新目标网络，避免频繁更新
            if update_counter % target_update_freq == 0:
              self.soft_update(self.actor_target, self.actor, self.tau)
              self.soft_update(self.critic_target, self.critic, self.tau)
          
        observation = observation_next
        
        # 避免过长的episode
        if steps >= 1000:  # LunarLander默认上限
          break
        
      # 记录回报
      self.return_traj.append(episode_return)
      self.recent_rewards.append(episode_return)
      if len(self.recent_rewards) > 100:  # 保持100个最近回报
        self.recent_rewards.pop(0)
      
      # 计算移动平均
      avg_return = np.mean(self.recent_rewards[-min(len(self.recent_rewards), 100):])
      
      # 记录平均损失
      if epoch_pi_losses:
        avg_pi_loss = np.mean(epoch_pi_losses)
        avg_critic_loss = np.mean(epoch_critic_losses)
        pi_losses.append(avg_pi_loss)
        critic_losses.append(avg_critic_loss)
        print(f"训练轮次: {epoch+1}/{self.epochs}, 回报: {episode_return:.2f}, 平均回报: {avg_return:.2f}, π损失: {avg_pi_loss:.4f}, Q损失: {avg_critic_loss:.4f}, 步数: {steps}")
      else:
        print(f"训练轮次: {epoch+1}/{self.epochs}, 回报: {episode_return:.2f}, 平均回报: {avg_return:.2f}, 步数: {steps}")
      # 保存最佳模型
      if episode_return > self.best_return:
        self.best_return = episode_return
        best_model_path = os.path.join(self.training_path, 'best_' + self.actor_filename)
        best_critic_path = os.path.join(self.training_path, 'best_' + self.critic_filename)
        torch.save(self.actor.state_dict(), best_model_path)
        torch.save(self.critic.state_dict(), best_critic_path)
        self.best_model_path = best_model_path
        print(f"💾 保存新的最佳模型 (回报: {self.best_return:.2f})")
      
      # 定期保存模型
      if ((epoch + 1) % self.save_freq == 0) or (epoch == self.epochs - 1):
        torch.save(self.actor.state_dict(), 
                  os.path.join(self.training_path, f'{epoch + 1}_{self.actor_filename}'))
        torch.save(self.critic.state_dict(), 
                  os.path.join(self.training_path, f'{epoch + 1}_{self.critic_filename}'))
        print(f"💾 保存第{epoch + 1}轮模型")
        
        # 绘制训练曲线
        self.plot_training_curves(pi_losses, critic_losses)
        
    # 保存回报历史到文件
    np.savetxt(os.path.join(self.training_path, 'ddpg_lunar_return_history.txt'), self.return_traj)
    
    # 最终绘制训练曲线
    self.plot_training_curves(pi_losses, critic_losses)
    print(f"训练完成，最佳回报: {self.best_return:.2f}")
    
  def plot_training_curves(self, pi_losses, critic_losses):
    """绘制训练曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    
    plt.figure(figsize=(12, 6))
    
    # 回报曲线
    plt.plot(self.return_traj, 'b-', alpha=0.7, label='每轮回报')
    
    if len(self.return_traj) >= 20:
      # 计算移动平均
      window = min(50, len(self.return_traj)//10)
      moving_avg = np.convolve(self.return_traj, np.ones(window)/window, mode='valid')
      plt.plot(range(window-1, len(self.return_traj)), moving_avg, 'r-', linewidth=2, label=f'移动平均({window})')
      
    plt.axhline(y=self.success_threshold, color='g', linestyle='--', label=f'成功线: {self.success_threshold}')
    plt.axhline(y=self.best_return, color='orange', linestyle='--', label=f'最佳: {self.best_return:.2f}')
    
    plt.title('DDPG on LunarLanderContinuous-v2 训练回报曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('回报值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(self.training_path, 'training_curve.png'), dpi=300)
    
  def ddpg_test(self, epochs=None):
    """测试训练好的模型"""
    if epochs is None:
      epochs = 10
      
    total_reward = 0
    print("\n开始测试DDPG模型...")
    
    for epoch in range(epochs):
      observation, _ = self.env.reset()
      episode_return = 0
      done = False
      steps = 0
      
      while not done:
        # 获取动作
        state = observation.reshape(1, -1)
        action = self.actor(torch.as_tensor(state, dtype=torch.float32)).detach().numpy()
        
        # 环境交互 (无噪声)
        observation_next, reward, terminated, truncated, _ = self.env.step(action[0])
        done = terminated or truncated
        
        # 累积回报
        episode_return += reward
        steps += 1
        observation = observation_next
        
      total_reward += episode_return
      print(f"测试 {epoch+1}/{epochs}, 回报: {episode_return:.2f}, 步数: {steps}")
      
    avg_reward = total_reward / epochs
    print(f"\n测试完成，平均回报: {avg_reward:.2f}")
    return avg_reward
    
  def load_model(self, training_path, actor_filename, critic_filename):
    """加载模型"""
    self.actor.load_state_dict(torch.load(os.path.join(training_path, actor_filename)))
    self.critic.load_state_dict(torch.load(os.path.join(training_path, critic_filename)))
    print(f"模型加载完成: {actor_filename}, {critic_filename}")

if __name__=='__main__':
  # 设置随机种子，确保结果可复现
  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)
  
  # 创建LunarLanderContinuous环境
  env_name = 'LunarLanderContinuous-v2'
  env = gym.make(env_name)
  
  # 创建DDPG智能体
  lunar_lander_ddpg = DDPG(env)
  
  # 训练模式
  lunar_lander_ddpg.ddpg_train()
  
  # 环境清理
  env.close()
  print("\n训练完成！")
