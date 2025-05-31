import gym
import torch
import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygame
from collections import deque

# Actor网络定义（必须与训练代码一致）
class Actor_Net(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, action_bounds=None):
        super(Actor_Net, self).__init__()
        self.act_dim = act_dim
        self.action_bounds = action_bounds
        #均值网络，利用前向神经网络
        self.actor_net = torch.nn.Sequential(
            layer_init(torch.nn.Linear(obs_dim, hidden_sizes[0])),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(hidden_sizes[1], act_dim), std=0.01),
            torch.nn.Tanh()
        )
        self.actor_net.requires_grad_()

    def forward(self, obs):
        # 输出动作，并根据动作边界进行缩放
        action_raw = self.actor_net(obs)
        if self.action_bounds is not None:
            action_low, action_high = self.action_bounds
            # 确保动作边界转换为张量并且与obs在同一设备上
            if isinstance(action_low, np.ndarray):
                action_low = torch.tensor(action_low, dtype=torch.float32, device=obs.device)
            elif isinstance(action_low, torch.Tensor) and action_low.device != obs.device:
                action_low = action_low.to(obs.device)
                
            if isinstance(action_high, np.ndarray):
                action_high = torch.tensor(action_high, dtype=torch.float32, device=obs.device)
            elif isinstance(action_high, torch.Tensor) and action_high.device != obs.device:
                action_high = action_high.to(obs.device)
            
            # 应用动作范围
            action = action_raw * 0.5 * (action_high - action_low) + 0.5 * (action_high + action_low)
        else:
            action = action_raw
        return action
    
    # 获取确定性动作，不计算梯度，测试时使用
    def get_a(self, obs):
        with torch.no_grad():
            action = self.forward(obs)
        return action

# 值网络定义（必须与训练代码一致）
class Critic_Net(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super(Critic_Net, self).__init__()
        self.layer_1 = layer_init(torch.nn.Linear(obs_dim, hidden_sizes[0]))
        self.layer_2 = layer_init(torch.nn.Linear(action_dim, hidden_sizes[0]))
        self.layer_3 = layer_init(torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.output = layer_init(torch.nn.Linear(hidden_sizes[1], 1))
    
    # 返回值函数预测值
    def forward(self, obs, action):
        q = self.layer_1(obs) + self.layer_2(action)
        q = torch.relu(self.layer_3(q))
        q = self.output(q)
        return q

# 初始化神经网络层权重
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def human_play_test(env, num_episodes=5):
    """方向键控制的人类玩家测试"""
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Lunar Lander Control - 按ESC退出")
    
    print("\n=== 方向键控制说明 ===")
    print("上方向键: 启动主引擎")
    print("左/右方向键: 控制侧向引擎")
    print("ESC键: 退出测试\n")
    
    scores = []
    try:
        for episode in range(1, num_episodes+1):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            clock = pygame.time.Clock()
            
            # 控制参数
            main_thrust = 0.0
            lateral_thrust = 0.0
            
            while not done:
                # 处理事件队列
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                      (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        return scores
                
                # 获取按键状态
                keys = pygame.key.get_pressed()
                
                # 主引擎控制（上键）
                main_thrust = 1.0 if keys[pygame.K_UP] else 0.0
                
                # 侧向控制（左右键）
                lateral_thrust = 0.0
                if keys[pygame.K_LEFT]:
                    lateral_thrust = 1.0
                if keys[pygame.K_RIGHT]:
                    lateral_thrust = -1.0
                
                # 构建动作向量 [主引擎，侧向推力]
                action = [main_thrust, lateral_thrust]
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
                
                # 渲染与帧率控制
                env.render()
                clock.tick(30)
            
            scores.append(total_reward)
            print(f"回合 {episode}/{num_episodes} 得分: {total_reward:.1f}")
            
    finally:
        pygame.quit()
    
    return scores

def model_test(env, model_path, num_episodes=5, render=True):
    """DDPG模型测试"""
    # 统一指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化Actor网络
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # 将动作边界转换为张量并指定设备
    action_bounds = [
        torch.tensor(env.action_space.low, dtype=torch.float32, device=device),
        torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
    ]
    
    hidden_sizes = [64, 64]
    
    actor = Actor_Net(obs_dim, act_dim, hidden_sizes, action_bounds).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    rewards = []
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 确保状态张量在正确的设备上
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # 直接使用前向传递获取动作
                action = actor(state_tensor)
            
            # 将动作转移到CPU进行环境交互
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            
            if render:
                env.render()
        
        rewards.append(episode_reward)
        if render or i == num_episodes - 1:  # 在渲染模式下或最后一轮显示进度
            print(f"Episode {i+1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    return {
        'mean': np.mean(rewards),
        'max': np.max(rewards),
        'min': np.min(rewards)
    }

def benchmark_test(model_dir, human_scores):
    """基准对比测试"""
    print("\n=== 基准对比测试 ===")
    
    # 收集模型文件
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith('_ddpg_lunar_actor.pth') and 
         f[0].isdigit()],  # 确保是以数字开头的模型文件 (如 100_ddpg_lunar_actor.pth)
        key=lambda x: int(re.search(r'(\d+)_ddpg_lunar_actor', x).group(1))
    )
    print(f"找到 {len(model_files)} 个模型文件")
    
    # 初始化环境
    env = gym.make('LunarLanderContinuous-v2')
    
    # 测试所有模型
    results = []
    progress = tqdm(model_files, desc="测试进度")
    for model_file in progress:
        try:
            model_path = os.path.join(model_dir, model_file)
            res = model_test(env, model_path, num_episodes=5, render=False)
            episode_num = int(re.search(r'(\d+)_ddpg_lunar_actor', model_file).group(1))
            results.append((episode_num, res))
        except Exception as e:
            print(f"测试模型 {model_file} 时发生错误: {e}")
            continue
    
    if not results:  # 检查是否有成功的测试结果
        print("没有成功的测试结果，无法生成对比图")
        env.close()
        return
    
    # 处理结果
    x = [r[0] for r in results]
    y_mean = [r[1]['mean'] for r in results]
    y_max = [r[1]['max'] for r in results]
    y_min = [r[1]['min'] for r in results]
    
    # 绘制对比图
    plt.figure(figsize=(15, 8))
    plt.plot(x, y_mean, 'b-', label='AI average')
    plt.fill_between(x, y_min, y_max, color='blue', alpha=0.1)
    plt.axhline(np.mean(human_scores), color='r', linestyle='--', 
                label=f'human baseline ({np.mean(human_scores):.1f}±{np.std(human_scores):.1f})')
    
    plt.title('DDPG vs Human on LunarLanderContinuous-v2')
    plt.xlabel('training episodes')
    plt.ylabel('score')
    plt.legend()
    plt.grid(True)
    
    # 保存结果
    plt.savefig('benchmark_comparison.png', dpi=300)
    print("\n对比图已保存为 benchmark_comparison.png")
    env.close()

def main():
    parser = argparse.ArgumentParser(description='DDPG月球着陆器测试脚本', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 标准测试模式
    parser_standard = subparsers.add_parser('standard', help='单模型测试')
    parser_standard.add_argument('--model_path', required=True, help='模型文件路径')
    parser_standard.add_argument('--num_episodes', type=int, default=5, 
                                 help='测试回合数')
    parser_standard.add_argument('--no_render', action='store_true', 
                                help='禁用渲染')

    # 基准测试模式
    parser_benchmark = subparsers.add_parser('benchmark', help='基准对比测试')
    parser_benchmark.add_argument('--model_dir', required=True, 
                                 help='包含模型文件的目录')

    # 人类测试模式
    parser_human = subparsers.add_parser('human', help='人类玩家模式')
    parser_human.add_argument('--num_episodes', type=int, default=5,
                             help='测试回合数')

    args = parser.parse_args()

    try:
        if args.command == 'standard':
            env = gym.make('LunarLanderContinuous-v2', 
                          render_mode='human' if not args.no_render else None)
            results = model_test(env, args.model_path, args.num_episodes, 
                                not args.no_render)
            env.close()
            
            print("\n测试结果:")
            print(f"平均得分: {results['mean']:.1f}")
            print(f"最高得分: {results['max']:.1f}")
            print(f"最低得分: {results['min']:.1f}")

        elif args.command == 'benchmark':
            env = gym.make('LunarLanderContinuous-v2', render_mode='human')
            print("=== 人类玩家测试 ===")
            human_scores = human_play_test(env, num_episodes=5)
            env.close()
            
            print("\n人类测试结果:")
            print(f"平均分: {np.mean(human_scores):.1f}")
            print(f"最高分: {np.max(human_scores):.1f}")
            print(f"最低分: {np.min(human_scores):.1f}")
            
            benchmark_test(args.model_dir, human_scores)

        elif args.command == 'human':
            env = gym.make('LunarLanderContinuous-v2', render_mode='human')
            scores = human_play_test(env, args.num_episodes)
            env.close()
            
            print("\n最终成绩:")
            print(f"平均分: {np.mean(scores):.1f}")
            print(f"最高分: {np.max(scores):.1f}")
            print(f"最低分: {np.min(scores):.1f}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        if 'env' in locals():
            env.close()

if __name__ == '__main__':
    main()