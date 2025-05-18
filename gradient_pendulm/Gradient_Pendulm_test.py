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

# 策略网络定义（必须与训练代码一致）
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mean_layer = torch.nn.Linear(256, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))
        return mean, self.log_std.exp()

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
    """AI模型测试"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化策略网络
    policy = PolicyNetwork(env.observation_space.shape[0], 
                          env.action_space.shape[0]).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                mean, _ = policy(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(mean.cpu().numpy())
            episode_reward += reward
            done = terminated or truncated
            state = next_state
        
        rewards.append(episode_reward)
    
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
        [f for f in os.listdir(model_dir) if f.endswith('.pth')],
        key=lambda x: int(re.search(r'model_(\d+).pth', x).group(1))
    )
    print(f"找到 {len(model_files)} 个模型文件")
    
    # 初始化环境
    env = gym.make('LunarLanderContinuous-v2')
    
    # 测试所有模型
    results = []
    progress = tqdm(model_files, desc="测试进度")
    for model_file in progress:
        model_path = os.path.join(model_dir, model_file)
        res = model_test(env, model_path, num_episodes=5, render=False)
        episode_num = int(re.search(r'model_(\d+).pth', model_file).group(1))
        results.append((episode_num, res))
    
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
    
    plt.title('AI vs human')
    plt.xlabel('training episodes')
    plt.ylabel('score')
    plt.legend()
    plt.grid(True)
    
    # 保存结果
    plt.savefig('benchmark_comparison.png', dpi=300)
    print("\n对比图已保存为 benchmark_comparison.png")
    env.close()

def main():
    parser = argparse.ArgumentParser(description='月球着陆器测试脚本', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 标准测试模式
    parser_standard = subparsers.add_parser('standard', help='单模型测试')
    parser_standard.add_argument('--model_path', required=True, help='模型文件路径')
    parser_standard.add_argument('--num_episodes', type=int, default=10, 
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