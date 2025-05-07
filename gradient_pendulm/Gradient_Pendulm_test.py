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
import time
#初始化神经网络层权重
def layer_init(layer,std = np.sqrt(2),bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight,std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer
#利用当前策略进行采样，产生数据
class Sample():
  def __init__(self,env):
    self.env = env
    self.gamma = 0.99
    self.lamda = 0.95
    self.last_state=None
    self.sum_reward = 0
    self.batch_state = None
    self.batch_act = None
    self.batch_logp = None
    self.batch_adv = None
    self.scale =1.25
    self.batch_val_target=None
    self.index = None
    self.episode_return = 0
    self.sum_return = 0
    self.sum_succ_episode = 0
    self.stop_action = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    self.x_traj=[]
    self.y_traj=[]
    self.action_traj_a1=[]
    self.action_traj_a2=[]
    self.action_traj_a3=[]
    self.action_traj_a4=[]
    self.action_traj_a5=[]
    self.action_traj_a6=[]
    self.action_traj_a7=[]
    self.action_traj_a8=[]
    self.action_traj_a9=[]
    self.action_traj_a10=[]
    self.phi_traj_1=[]
    self.phi_traj_2=[]
    self.phi_traj_3=[]
    self.phi_traj_4=[]
    self.phi_traj_5=[]
    self.phi_traj_6=[]
    self.phi_traj_7=[]
    self.phi_traj_8=[]
    self.phi_traj_9=[]
    self.phi_traj_10=[]
  #以policy_net网络进行采样1条轨迹,返回构建动作网络损失函数所需要的数据obs, log_as, adv，构建值网络所需要的数据：val_target
  def sample_one_episode(self, actor_net):
    #产生num_episodes条轨迹
    Flag = 0
    val_target=0
    episode_obs = []
    episode_vals=[]
    episode_actions=[]
    episode_rewards = []
    episode_val_target=[]
    done = False
    num_episodes = 1
    episode_sum = 0
    # print("当前目标：",self.env.target)
    for i in range(num_episodes):
      cur_obs = self.env.reset()
      steps = 0
      while True:
        episode_obs.append(cur_obs)
        #采样动作，及动作的对数，不计算梯度
        action,log_a = actor_net.get_a(torch.as_tensor(cur_obs,dtype=torch.float32))
        episode_actions.append(action)
        episode_val_target.append(0.0)
        #往前推进一步
        # for j in range(40):
        next_obs,reward,done,_ = self.env.step(action)
        # self.env.render()
        cur_obs = next_obs
        episode_rewards.append(reward)
        steps=steps+1
        #处理回报
        if(done):
          self.last_state = next_obs
          val_target = 0.0
          episode_vals.append(val_target)
          # print("val_target:",val_target)
          discounted_sum_reward = np.zeros_like(episode_rewards)
          #计算mbatch折扣累积回报
          # print("总长度：",len(episode_rewards))
          for t in reversed(range(0,len(episode_rewards))):
            val_target = episode_rewards[t]+val_target*self.gamma
            episode_val_target[t] = val_target
            episode_sum+=episode_rewards[t]
          break
      #需要转换的数据为batch_obs, batch_act,batch_logp, batch_adv,batch_val_target
      episode_obs = np.reshape(episode_obs,[len(episode_obs),3])
      episode_actions = np.reshape(episode_actions,[len(episode_actions),1])
      #目标值函数
      episode_val_target = np.reshape(episode_val_target,[len(episode_val_target),1])
    return episode_obs, episode_actions, episode_val_target,episode_sum
  def test_sample_one_episode(self, actor_net):
    #产生num_episodes条轨迹
    Flag = 0
    val_target=0
    episode_obs = []
    episode_vals=[]
    episode_actions=[]
    episode_rewards = []
    episode_val_target=[]
    done = False
    num_episodes = 1
    episode_sum = 0
    # print("当前目标：",self.env.target)
    for i in range(num_episodes):
      cur_obs = self.env.reset()[0]
      steps = 0
      while True:
        episode_obs.append(cur_obs)
        #采样动作，及动作的对数，不计算梯度
        action,log_a = actor_net.get_a(torch.as_tensor(cur_obs,dtype=torch.float32))
        episode_actions.append(action.item())
        episode_val_target.append(0.0)
        #往前推进一步
        # for j in range(40):
        next_obs,reward,done,_,_ = self.env.step(action)
        #print(action)
        #渲染环境
        self.env.render()
        # time.sleep(0.1)
        cur_obs = next_obs
        episode_rewards.append(reward)
        steps=steps+1
        #处理回报
        if(done):
          self.last_state = next_obs
          val_target = 0.0
          episode_vals.append(val_target)
          print("动作序列为：",episode_actions)
          # print("val_target:",val_target)
          discounted_sum_reward = np.zeros_like(episode_rewards)
          #计算mbatch折扣累积回报
          # print("总长度：",len(episode_rewards))
          for t in reversed(range(0,len(episode_rewards))):
            val_target = episode_rewards[t]+val_target*self.gamma
            episode_val_target[t] = val_target
            episode_sum+=episode_rewards[t]
          print("测试总回报为：",episode_sum)
          break
      #需要转换的数据为batch_obs, batch_act,batch_logp, batch_adv,batch_val_target
      episode_obs = np.reshape(episode_obs,[len(episode_obs),3])
      episode_actions = np.reshape(episode_actions,[len(episode_actions),1])
      #目标值函数
      episode_val_target = np.reshape(episode_val_target,[len(episode_val_target),1])
    # return episode_obs, episode_actions, episode_val_target,episode_sum
  def sample_many_episodes(self,actor_net,num):
    self.sum_return = 0
    self.Flag = 0
    episode_sum = 0
    batch_state,batch_act,batch_val_target,episode_sum=self.sample_one_episode(actor_net)
    self.sum_return = self.episode_return+episode_sum
    for i in range(num):
      episode_state,episode_act,episode_val_target,episode_sum=self.sample_one_episode(actor_net)
      batch_state=np.concatenate((batch_state,episode_state),0)
      batch_act = np.concatenate((batch_act,episode_act),0)
      batch_val_target = np.concatenate((batch_val_target,episode_val_target),0)
      self.sum_return = self.sum_return+episode_sum
    self.batch_state = batch_state
    self.batch_act = batch_act
    self.batch_val_target = batch_val_target
    # print("batch_act",batch_act)
    # print("元素数量：",self.batch_adv.shape[0])
  def get_data(self,start_index,sgd_num):
    sgd_batch_state = np.zeros((sgd_num,3))
    sgd_batch_act = np.zeros((sgd_num,1))
    sgd_batch_val_target=np.zeros((sgd_num,1))
    for i in range(sgd_num):
      sgd_batch_state[i,:] = self.batch_state[self.index[start_index+i],:]
      sgd_batch_act[i,:] = self.batch_act[self.index[start_index+i],:]
      sgd_batch_val_target[i,:] = self.batch_val_target[self.index[start_index+i],:]
    #在minibatch中归一化sgd_batch_val
    sgd_batch_val = (sgd_batch_val_target-sgd_batch_val_target.mean())/sgd_batch_val_target.std()
    # print("随机优势值函数：",sgd_batch_adv)
    #将numpy数据转化为torch数据
    sgd_batch_state = torch.as_tensor(sgd_batch_state,dtype=torch.float32)
    sgd_batch_act = torch.as_tensor(sgd_batch_act,dtype=torch.float32)
    sgd_batch_val = torch.as_tensor(sgd_batch_val, dtype=torch.float32)
    return sgd_batch_state,sgd_batch_act,sgd_batch_val

#构建策略网络
class Actor_Net(nn.Module):
  def __init__(self,obs_dim,act_dim,hidden_sizes):
    super(Actor_Net,self).__init__()
    self.act_dim = act_dim
    #标准差的对数
    log_std = -0.5*np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    # self.log_std = torch.as_tensor(log_std)
    #均值和标准差网络，利用前向神经网路
    self.actor_net = nn.Sequential(
      layer_init(nn.Linear(obs_dim,hidden_sizes[0])),
      nn.ReLU(),
      layer_init(nn.Linear(hidden_sizes[0],hidden_sizes[1])),
      nn.ReLU(),
      layer_init(nn.Linear(hidden_sizes[1],act_dim),std =1.0),
      nn.Tanh()
    )
    self.actor_net.requires_grad_()
  #计算分布
  def distribution(self, obs):
    mu = self.actor_net(obs)
    log_std = self.log_std.expand_as(mu)
    std = torch.exp(log_std)
    return Normal(mu, std)
  #计算分布的对数
  def log_prob_from_distribution(self, dist, act):
    return dist.log_prob(act).sum()
  #批策略对数
  def _log_prob_from_distribution(self,dist,act):
    return dist.log_prob(act).sum(axis=1)
  #返回概率分布及动作act的概率，计算梯度，计算动作损失函数时使用
  def forward(self, obs, act=None):
    dist = self.distribution(obs)
    logp_a = None
    if act is not None:
      logp_a = self._log_prob_from_distribution(dist,act)
    return dist, logp_a, dist.entropy().sum(axis = 1)
  #采样一个动作，不计算梯度,产生数据样本时使用
  def get_a(self,obs):
    with torch.no_grad():
      dist = self.distribution(obs)
      action = dist.sample()
      #截取（-1,1）的值
      action = torch.clip(action, -2,2)
      log_a = self.log_prob_from_distribution(dist,action)
      # print("action:", action)
    return action.numpy(),log_a.numpy()
  def get_mu(self,obs):
    with torch.no_grad():
      mu = self.actor_net(obs)
    return mu.numpy()
#Policy_Gradient算法类
class Policy_Gradient():
    def __init__(self,env):
      self.env = env
      self.sampler = Sample(env)
      self.obs_dim = 3
      self.act_dim = 1
      self.hidden = [32,32]
      self.pi_lr=0.00004
      self.train_pi_iters = 10
      self.sgd_num = 256
      self.epochs = 2000
      self.save_freq = 100
      self.episodes_num = 5
      self.target_kl = 1.0
      self.return_traj = []
      self.succ_rate_traj = []
      self.actor = Actor_Net(self.obs_dim,self.act_dim,self.hidden)
      self.pi_optimizer =Adam(self.actor.parameters(),self.pi_lr)
      self.training_path = 'D:\\00 第三本书\\code'
      self.actor_filename = 'pg_actor.pth'
    #计算策略损失函数
    def compute_loss_pi(self,obs, act,value):
      act_dist,logp,entropy = self.actor(obs,act)
      num = obs.size()[0]
      # torch.reshape(logp,(num,1))修改维度保持一致
      logp = logp.reshape(num, 1)
      loss_pi = -(logp*value).mean()
      # print("loss:",loss_pi)
      loss_entropy = entropy.mean()
      return loss_pi,loss_entropy
    #采集数据，并进行更新
    def update(self):
      #指定目标位置，采集数据,得到批数据
      #batch_state, batch_act, batch_logp, batch_adv, batch_val_target=self.sampler.sample_one_episodes(self.actor,self.critic)
      self.sampler.sample_many_episodes(self.actor,self.episodes_num)
      self.return_traj.append(self.sampler.sum_return/6)
      # self.succ_rate_traj.append(self.sampler.sum_succ_episode/101)
      print("当前回报：",self.sampler.sum_return/6)
      np.savetxt('return_traj.txt', self.return_traj)
      # np.savetxt('succ_rate.txt',self.succ_rate_traj)
      batch_size = self.sampler.batch_state.shape[0]
      # print("batch_size",batch_size)
      self.sampler.index = np.arange(batch_size)
      loss_num = 0.0
      #利用采集的数据训练策略网络
      for i in range(self.train_pi_iters):
        np.random.shuffle(self.sampler.index)
        approx_kl = 0.0
        for start_index in range(0,batch_size-self.sgd_num,self.sgd_num):
          #采集mini批数据，计算随机梯度
          # print("start_index:",start_index)
          batch_state, batch_act, batch_val_target =self.sampler.get_data(start_index,self.sgd_num)
          #1.清除梯度值
          self.pi_optimizer.zero_grad()
          #2. 构建损失函数
          loss_pi,loss_entropy= self.compute_loss_pi(batch_state,batch_act,batch_val_target)
          loss = loss_pi-0.001*loss_entropy
          #loss = loss_pi + 0.5 * loss_critic
          # print("loss:",loss)
          #3. 反向传播梯度
          loss.backward()
          #限制最大梯度
          nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
          #4. 参数更新一步
          self.pi_optimizer.step()
          # print("更新后标准差：", self.actor.log_std)
        if(i==0):
          print("标准差：",self.actor.log_std)
    def pg_train(self,epochs=None):
      self.epochs = epochs
      for epoch in range(self.epochs):
        print("训练次数:",epoch)
        self.update()
        if ((epoch + 1) % self.save_freq == 0):
          # 每训练N次保存模型
          torch.save(self.actor.state_dict(), os.path.join(self.training_path, self.actor_filename + str(epoch)))
          # #探索衰减
          self.actor.log_std-=0.01
      plt.plot(self.return_traj)
      plt.show()
    def load_model(self,training_path,actor_filename):
      self.actor.load_state_dict(torch.load(os.path.join(training_path, actor_filename)))
    def test_one_episode(self):
      #指定目标位置，采集数据,得到批数据
      self.env.Render = True
      self.sampler.test_sample_one_episode(self.actor)

if __name__=='__main__':
  # 固定随机种子
  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  #构建单摆类
  env_name = 'Pendulum-v1'
  env = gym.make(env_name,render_mode="human")
  env.reset()
  #env.unwrapped
  env.render()
#  env.seed(1)
  print(env.action_space.high,env.state)
  #定义力矩取值区间
  action_bound = [-env.action_space.high, env.action_space.high]
  pendulum_pg = Policy_Gradient(env)
  # pendulum_pg.pg_train(10000)
  return_traj = np.loadtxt('return_traj_1.txt')
  plt.plot(return_traj)
  plt.show()
  training_path = 'E:\\book\\code\\pg_pendulm'
  actor_filename = 'pg_actor.pth' + "5999"
  pendulum_pg.load_model(training_path, actor_filename)
  # log_std = -2.5 * np.ones(pendulum_pg.act_dim, dtype=np.float32)
  # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
  # pendulum_pg.log_std = torch.as_tensor(log_std)
  pendulum_pg.test_one_episode()
  env.reset()


  while(1):
      # action = np.array([0])
      # env.step(action)
      pendulum_pg.test_one_episode()

