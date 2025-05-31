## README
### 文件结构介绍
`./models/`  # 模型文件
`ddpg_Pendulm.py` # 训练脚本 
`ddpg_LunarLander_test.py` # 测试脚本
`Human.gif` # 人类玩家进行测试演示
`Model.gif` # 训练好的模型进行测试演示
`training_curve.png` # 训练曲线
`benchmark_comparison.png` # 人类玩家与训练好的模型的对比

### 使用指南
#### 训练模型
`python ddpg_Pendulm.py`
#### 测试模型
- 测试训练好的模型文件
`pythonddpg_LunarLander_test.py standard --model_path \models\best_ddpg_lunar_actor.pth --num_episodes 5`
- 进行人类玩家测试
`python ddpg_LunarLander_test.py human --num_episodes 5`
- 测试所有的模型文件并绘制基准对比曲线
`python ddpg_LunarLander_test.py benchmark --model_dir ./models`

