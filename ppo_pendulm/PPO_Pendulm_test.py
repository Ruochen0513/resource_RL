import gym
import torch
import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import deque

# 复制自训练脚本
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor_Net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, action_bounds=None):
        super(Actor_Net, self).__init__()
        self.act_dim = act_dim
        self.action_bounds = action_bounds
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], act_dim), std=0.01),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, obs, act=None):
        mu = self.shared(obs)
        if self.action_bounds is not None:
            action_low, action_high = self.action_bounds
            mu = torch.tanh(mu) * 0.5 * (action_high - action_low) + 0.5 * (action_high + action_low)
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = torch.distributions.Normal(mu, std)
        if act is not None:
            logp = dist.log_prob(act).sum(dim=-1)
        else:
            logp = None
        entropy = dist.entropy().sum(dim=-1) if self.act_dim > 1 else dist.entropy()
        return dist, logp, entropy
    def get_mean_action(self, obs):
        with torch.no_grad():
            mu = self.shared(obs)
            if self.action_bounds is not None:
                action_low, action_high = self.action_bounds
                mu = torch.tanh(mu) * 0.5 * (action_high - action_low) + 0.5 * (action_high + action_low)
            return mu.cpu().numpy()

# 标准AI测试

def model_test(env, actor, num_episodes=5, render=True):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            obs_tensor = torch.as_tensor(state, dtype=torch.float32)
            action = actor.get_mean_action(obs_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            if render:
                env.render()
        rewards.append(episode_reward)
    return {
        'mean': np.mean(rewards),
        'max': np.max(rewards),
        'min': np.min(rewards)
    }

# 人类测试（键盘控制）
def human_play_test(env, num_episodes=5):
    try:
        import pygame
    except ImportError:
        print("未安装pygame，无法进行人类测试模式。")
        return []
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("LunarLanderContinuous Control - 按ESC退出")
    print("\n=== LunarLanderContinuous 控制说明 ===")
    print("上/下方向键: 主引擎推力 (上=最大, 下=最小)")
    print("左/右方向键: 侧推力 (左=左推, 右=右推)")
    print("ESC键: 退出测试\n")
    scores = []
    try:
        for episode in range(1, num_episodes+1):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            clock = pygame.time.Clock()
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                      (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        return scores
                keys = pygame.key.get_pressed()
                action = np.array([0.0, 0.0], dtype=np.float32)
                # 主引擎推力
                if keys[pygame.K_UP]:
                    action[0] = 1.0
                elif keys[pygame.K_DOWN]:
                    action[0] = -1.0
                else:
                    action[0] = 0.0
                # 侧推力
                if keys[pygame.K_LEFT]:
                    action[1] = -1.0
                elif keys[pygame.K_RIGHT]:
                    action[1] = 1.0
                else:
                    action[1] = 0.0
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                state = next_state
                env.render()
                clock.tick(30)
            scores.append(total_reward)
            print(f"回合 {episode}/{num_episodes} 得分: {total_reward:.1f}")
    finally:
        pygame.quit()
    return scores

# 基准对比测试（只遍历*_actor.pth，按数字和best排序）
def extract_num(filename):
    match = re.match(r'(\d+)_actor\.pth', filename)
    if match:
        return int(match.group(1))
    elif filename == 'best_actor.pth':
        return float('inf')
    else:
        return -1

def benchmark_test(model_dir, env, human_scores):
    print("\n=== 基准对比测试 ===")
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith('_actor.pth')],
        key=extract_num
    )
    print(f"找到 {len(model_files)} 个模型文件")
    results = []
    for model_file in model_files:
        print(f"加载模型文件: {model_file}")
        model_path = os.path.join(model_dir, model_file)
        actor = load_actor(model_path, env)
        res = model_test(env, actor, num_episodes=5, render=False)
        # 横坐标支持best
        if model_file == 'best_actor.pth':
            episode_num = 'best'
        else:
            m = re.match(r'(\d+)_actor\.pth', model_file)
            episode_num = int(m.group(1)) if m else -1
        results.append((episode_num, res))
    # 按数字（和best）排序
    results.sort(key=lambda x: (float('inf') if x[0] == 'best' else x[0]))
    x = [r[0] for r in results]
    y_mean = [r[1]['mean'] for r in results]
    y_max = [r[1]['max'] for r in results]
    y_min = [r[1]['min'] for r in results]
    plt.figure(figsize=(15, 8))
    # 横坐标处理：数字+best
    x_labels = [str(i) for i in x]
    plt.plot(x_labels, y_mean, 'b-', label='AI average')
    plt.fill_between(x_labels, y_min, y_max, color='blue', alpha=0.1)
    if human_scores:
        plt.axhline(np.mean(human_scores), color='r', linestyle='--', 
                    label=f'human baseline ({np.mean(human_scores):.1f}±{np.std(human_scores):.1f})')
    plt.title('AI vs human')
    plt.xlabel('training episodes')
    plt.ylabel('score')
    plt.legend()
    plt.grid(True)
    # 只显示100的倍数和best
    xticks = []
    xticklabels = []
    for i, label in enumerate(x_labels):
        if label == 'best' or (label.isdigit() and int(label) % 200 == 0):
            xticks.append(i)
            xticklabels.append(label)
    plt.xticks(xticks, xticklabels)
    plt.savefig('benchmark_comparison.png', dpi=300)
    print("\n对比图已保存为 benchmark_comparison.png")

# 加载actor模型
def load_actor(model_path, env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_bounds = [
        torch.as_tensor(env.action_space.low, dtype=torch.float32),
        torch.as_tensor(env.action_space.high, dtype=torch.float32)
    ]
    actor = Actor_Net(obs_dim, act_dim, [64, 64], action_bounds)
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    actor.eval()
    return actor

def main():
    parser = argparse.ArgumentParser(description='PPO Pendulum 测试脚本', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    parser_standard = subparsers.add_parser('standard', help='单模型测试')
    parser_standard.add_argument('--model_path', required=True, help='模型文件路径')
    parser_standard.add_argument('--num_episodes', type=int, default=5, help='测试回合数')
    parser_standard.add_argument('--no_render', action='store_true', help='禁用渲染')
    parser_benchmark = subparsers.add_parser('benchmark', help='基准对比测试')
    parser_benchmark.add_argument('--model_dir', required=True, help='包含模型文件的目录')
    parser_human = subparsers.add_parser('human', help='人类玩家模式')
    parser_human.add_argument('--num_episodes', type=int, default=5, help='测试回合数')
    args = parser.parse_args()
    try:
        if args.command == 'standard':
            env = gym.make('LunarLanderContinuous-v2', render_mode='human' if not args.no_render else None)
            actor = load_actor(args.model_path, env)
            results = model_test(env, actor, args.num_episodes, not args.no_render)
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
            env = gym.make('LunarLanderContinuous-v2')
            benchmark_test(args.model_dir, env, human_scores)
            env.close()
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