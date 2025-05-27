import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import os
import time
from collections import deque


class Sample:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99  # 折扣因子
        self.lamda = 0.95  # GAE参数
        self.batch_state = None
        self.batch_act = None
        self.batch_logp = None
        self.batch_adv = None
        self.batch_val_target = None
        self.index = None
        self.sum_return = 0
        
    def sample_one_episode(self, actor_net, critic_net):
        """采样一条轨迹"""
        episode_obs = []
        episode_actions = []
        episode_logps = []
        episode_rewards = []
        episode_vals = []
        cur_obs, _ = self.env.reset()
        done = False
        episode_sum = 0
        while not done:
            episode_obs.append(cur_obs)
            obs_tensor = torch.as_tensor(cur_obs, dtype=torch.float32)
            # 采样连续动作和log概率
            action, logp = actor_net.get_a(obs_tensor)
            value = critic_net.get_v(obs_tensor)
            episode_actions.append(action)
            episode_logps.append(logp)
            episode_vals.append(value)
            # 环境交互
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            cur_obs = next_obs
            episode_rewards.append(reward)
            episode_sum += reward
        
        # GAE优势计算
        vals = episode_vals + [0]  # 末状态价值设为0
        episode_adv = []
        adv = 0
        
        for t in reversed(range(len(episode_rewards))):
            delta = episode_rewards[t] + self.gamma * vals[t+1] - vals[t]
            adv = delta + self.gamma * self.lamda * adv
            episode_adv.insert(0, adv)
        
        # 目标值函数（累计折扣回报）
        val_target = []
        ret = 0
        for r in reversed(episode_rewards):
            ret = r + self.gamma * ret
            val_target.insert(0, ret)
            
        return (np.array(episode_obs), np.array(episode_actions),
                np.array(episode_logps).reshape(-1, 1), np.array(episode_adv).reshape(-1, 1),
                np.array(val_target).reshape(-1, 1), episode_sum)
    
    def sample_many_episodes(self, actor_net, critic_net, num):
        """采样多条轨迹"""
        self.sum_return = 0
        
        # 采集第一条轨迹
        batch_state, batch_act, batch_logp, batch_adv, batch_val_target, episode_sum = self.sample_one_episode(actor_net, critic_net)
        self.sum_return += episode_sum
        
        # 采集剩余轨迹
        for i in range(num - 1):
            episode_state, episode_act, episode_logp, episode_adv, episode_val_target, episode_sum = self.sample_one_episode(actor_net, critic_net)
            batch_state = np.concatenate((batch_state, episode_state), 0)
            batch_act = np.concatenate((batch_act, episode_act), 0)
            batch_logp = np.concatenate((batch_logp, episode_logp), 0)
            batch_adv = np.concatenate((batch_adv, episode_adv), 0)
            batch_val_target = np.concatenate((batch_val_target, episode_val_target), 0)
            self.sum_return += episode_sum
            
        self.batch_state = batch_state
        self.batch_act = batch_act
        self.batch_logp = batch_logp
        self.batch_adv = batch_adv
        self.batch_val_target = batch_val_target
    
    def get_data(self, start_index, sgd_num):
        """获取mini-batch数据"""
        idx = self.index[start_index:start_index+sgd_num]
        sgd_batch_state = self.batch_state[idx]
        sgd_batch_act = self.batch_act[idx]
        sgd_batch_logp = self.batch_logp[idx]
        sgd_batch_adv = self.batch_adv[idx]
        sgd_batch_val_target = self.batch_val_target[idx]
        
        # 归一化优势
        if sgd_batch_adv.std() > 1e-8:
            sgd_batch_adv = (sgd_batch_adv - sgd_batch_adv.mean()) / (sgd_batch_adv.std() + 1e-8)
        else:
            sgd_batch_adv = sgd_batch_adv - sgd_batch_adv.mean()
            
        # 转换为tensor
        sgd_batch_state = torch.as_tensor(sgd_batch_state, dtype=torch.float32)
        sgd_batch_act = torch.as_tensor(sgd_batch_act, dtype=torch.float32)  
        sgd_batch_logp = torch.as_tensor(sgd_batch_logp, dtype=torch.float32)
        sgd_batch_adv = torch.as_tensor(sgd_batch_adv, dtype=torch.float32)
        sgd_batch_val_target = torch.as_tensor(sgd_batch_val_target, dtype=torch.float32)
        
        return sgd_batch_state, sgd_batch_act, sgd_batch_logp, sgd_batch_adv, sgd_batch_val_target


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """神经网络层初始化"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor_Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, action_bounds=None):
        super(Actor_Net, self).__init__()
        self.act_dim = act_dim
        self.action_bounds = action_bounds  # 动作范围
        
        # 共享特征提取网络
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], act_dim), std=0.01),
            nn.Tanh()
        )
        
        
        # 标准差网络
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, obs, act=None):
        mu = self.shared(obs)
        
        # 应用动作边界
        if self.action_bounds is not None:
            # 缩放动作到边界范围
            action_low, action_high = self.action_bounds
            mu = torch.tanh(mu) * 0.5 * (action_high - action_low) + 0.5 * (action_high + action_low)
        
        # 获取标准差并限制范围
        std = torch.exp(self.log_std.clamp(-20, 2))
        
        # 创建正态分布
        dist = Normal(mu, std)
        
        if act is not None:
            logp = dist.log_prob(act).sum(dim=-1)  # 连续动作每个维度的log_prob求和
        else:
            logp = None
            
        entropy = dist.entropy().sum(dim=-1) if self.act_dim > 1 else dist.entropy()
        return dist, logp, entropy
    
    def get_a(self, obs):
        """获取连续动作和log概率"""
        with torch.no_grad():
            dist, _, _ = self.forward(obs)
            action = dist.sample()
            logp = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), logp.item()


class Critic_Net(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super(Critic_Net, self).__init__()
        # 针对LunarLanderContinuous优化：使用Tanh激活函数
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0)
        )
        
    def forward(self, obs):
        return self.critic_net(obs).squeeze()
    
    def get_v(self, obs):
        """获取状态价值"""
        with torch.no_grad():
            return self.forward(obs).item()


class PPO:
    def __init__(self, env, hidden_sizes, pi_lr, critic_lr):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]  # 连续动作空间维度
        
        # 获取动作范围并转换为张量
        self.action_bounds = [
            torch.as_tensor(env.action_space.low, dtype=torch.float32),
            torch.as_tensor(env.action_space.high, dtype=torch.float32)
        ]

        # 网络初始化
        self.actor = Actor_Net(self.obs_dim, self.act_dim, hidden_sizes, self.action_bounds)
        self.critic = Critic_Net(self.obs_dim, hidden_sizes)
        # 优化器
        self.pi_optimizer = Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        # PPO超参数
        self.clip_ratio = 0.2
        self.epochs = 4000
        self.episodes_num = 10  # 每次更新采集的轨迹数
        self.train_pi_iters = 80
        self.sgd_num = 64  # 批大小
        self.save_freq = 10
        # 训练监控
        self.sampler = Sample(env)
        self.return_traj = []
        self.policy_losses = []
        self.entropy_losses = []
        self.critic_losses = []
        self.kl_divergences = []
        self.recent_rewards = deque(maxlen=100)  # 用于计算移动平均
        # 模型保存
        self.training_path = r'D:\\code\\resource_RL\\ppo_pendulm\\models'
        os.makedirs(self.training_path, exist_ok=True)
        self.best_return = -float('inf')
        self.best_model_path = None
        # 早停机制
        self.early_stop_kl = 0.02  # KL散度阈值
        self.success_threshold = 200  # 模型训练成功标准
        
    def compute_loss_pi(self, obs, act, logp_old, adv):
        """计算策略损失"""
        dist, logp, entropy = self.actor(obs, act)
        ratio = torch.exp(logp - logp_old.squeeze())
        
        # PPO-clip损失
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv.squeeze()
        loss_pi = -torch.min(ratio * adv.squeeze(), clip_adv).mean()
        
        # 熵正则化
        loss_entropy = entropy.mean()
        
        # KL散度估计
        approx_kl = (logp_old.squeeze() - logp).mean().item()
        
        return loss_pi, loss_entropy, approx_kl
    
    def compute_loss_critic(self, obs, val_target):
        """计算价值函数损失"""
        return ((self.critic(obs) - val_target.squeeze()) ** 2).mean()
    
    def update(self):
        """PPO更新步骤"""
        # 采集数据
        self.sampler.sample_many_episodes(self.actor, self.critic, self.episodes_num)
        avg_return = self.sampler.sum_return / self.episodes_num
        self.return_traj.append(avg_return)
        self.recent_rewards.append(avg_return)
        
        batch_size = self.sampler.batch_state.shape[0]
        self.sampler.index = np.arange(batch_size)
        
        # 累积损失用于监控
        sum_pi_loss = 0.0
        sum_entropy_loss = 0.0
        sum_critic_loss = 0.0
        sum_kl = 0.0
        update_count = 0
        
        # 训练循环
        for i in range(self.train_pi_iters):
            np.random.shuffle(self.sampler.index)
            
            for start_index in range(0, batch_size - self.sgd_num, self.sgd_num):
                batch_state, batch_act, batch_logp, batch_adv, batch_val_target = self.sampler.get_data(start_index, self.sgd_num)
                
                # 训练策略网络
                self.pi_optimizer.zero_grad()
                loss_pi, loss_entropy, approx_kl = self.compute_loss_pi(batch_state, batch_act, batch_logp, batch_adv)
                total_pi_loss = loss_pi - 0.01 * loss_entropy 
                total_pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.pi_optimizer.step()
                
                # 训练价值网络
                self.critic_optimizer.zero_grad()
                loss_critic = self.compute_loss_critic(batch_state, batch_val_target)
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                # 累积统计
                sum_pi_loss += loss_pi.item()
                sum_entropy_loss += loss_entropy.item()
                sum_critic_loss += loss_critic.item()
                sum_kl += approx_kl
                update_count += 1
                
                # 早停检查
                if approx_kl > self.early_stop_kl:
                    print(f"KL散度过大({approx_kl:.4f})，提前停止训练")
                    break
            
            if sum_kl / max(update_count, 1) > self.early_stop_kl:
                break
        
        # 记录平均损失
        if update_count > 0:
            self.policy_losses.append(sum_pi_loss / update_count)
            self.entropy_losses.append(sum_entropy_loss / update_count)
            self.critic_losses.append(sum_critic_loss / update_count)
            self.kl_divergences.append(sum_kl / update_count)
        
        moving_avg = np.mean(self.recent_rewards) if self.recent_rewards else avg_return
        print(f"平均回报: {avg_return:.2f}, 近100轮平均: {moving_avg:.2f}")
        
        # 检查是否达成功
        if moving_avg >= self.success_threshold:
            print(f"🎉 达到成功标准！近100轮平均回报: {moving_avg:.2f}")
    
    def ppo_train(self):
        """主训练循环"""
        start_time = time.time()
        success_count = 0
        
        print("开始LunarLanderContinuous PPO训练...")
        print(f"目标：回报达到 {self.success_threshold}")
        
        for epoch in range(self.epochs):
            print(f"\n=== 训练轮次 {epoch+1}/{self.epochs} ===")
            self.update()
            
            # 保存最佳模型
            current_return = self.return_traj[-1]
            if current_return > self.best_return:
                self.best_return = current_return
                best_model_path = os.path.join(self.training_path, 'best_actor.pth')
                best_critic_path = os.path.join(self.training_path, 'best_critic.pth')
                torch.save(self.actor.state_dict(), best_model_path)
                torch.save(self.critic.state_dict(), best_critic_path)
                self.best_model_path = best_model_path
                print(f"💾 保存新的最佳模型 (回报: {self.best_return:.2f})")
            
            # 定期保存
            if (epoch + 1) % self.save_freq == 0:
                torch.save(self.actor.state_dict(), os.path.join(self.training_path, f'{epoch + 1}_actor.pth'))
                torch.save(self.critic.state_dict(), os.path.join(self.training_path, f'{epoch + 1}_critic.pth'))
                print(f"💾 保存第{epoch + 1}轮模型")
        
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\n训练完成，耗时: {training_duration:.2f}秒")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文黑体
        plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 回报曲线
        axes[0, 0].plot(self.return_traj, 'b-', alpha=0.7)
        if len(self.return_traj) >= 20:
            # 计算移动平均
            window = min(50, len(self.return_traj)//10)
            moving_avg = np.convolve(self.return_traj, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.return_traj)), moving_avg, 'r-', linewidth=2, label=f'移动平均({window})')
        axes[0, 0].axhline(y=self.success_threshold, color='g', linestyle='--', label=f'成功线: {self.success_threshold}')
        axes[0, 0].axhline(y=self.best_return, color='orange', linestyle='--', label=f'最佳: {self.best_return:.2f}')
        axes[0, 0].set_title('训练回报曲线')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('平均回报')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 策略损失
        if self.policy_losses:
            axes[0, 1].plot(self.policy_losses, 'r-')
            axes[0, 1].set_title('策略损失')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('损失')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 价值损失
        if self.critic_losses:
            axes[1, 0].plot(self.critic_losses, 'b-')
            axes[1, 0].set_title('价值网络损失')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('损失')
            axes[1, 0].grid(True, alpha=0.3)
        
        # KL散度
        if self.kl_divergences:
            axes[1, 1].plot(self.kl_divergences, 'g-')
            axes[1, 1].axhline(y=self.early_stop_kl, color='r', linestyle='--', label=f'早停线: {self.early_stop_kl}')
            axes[1, 1].set_title('KL散度')
            axes[1, 1].set_xlabel('轮次')
            axes[1, 1].set_ylabel('KL散度')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.training_path, 'training_curves.png'), dpi=300)
        plt.show()
    
    def load_model(self, training_path, actor_filename, critic_filename):
        """加载模型"""
        self.actor.load_state_dict(torch.load(os.path.join(training_path, actor_filename)))
        self.critic.load_state_dict(torch.load(os.path.join(training_path, critic_filename)))
        print(f"模型加载完成: {actor_filename}, {critic_filename}")


if __name__ == '__main__':
    # 创建环境
    env = gym.make('LunarLanderContinuous-v2')
    # 创建PPO智能体
    lunarlander_ppo = PPO(env, 
                         hidden_sizes=[64, 64], 
                         pi_lr=3e-4,             # 策略学习率
                         critic_lr=1e-3)         # 价值网络学习率
    # 开始训练
    lunarlander_ppo.ppo_train()
    
    # 环境清理
    env.close()
    print("\n训练完成！")
