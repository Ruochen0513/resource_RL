## <center>强化学习实验第一次实验报告
#### <center> 2213530 张禹豪 智能科学与技术
### 一、实验要求
- 将已有的3x3的井字棋强化学习代码扩展为4x4的井字棋
- 分别训练迭代不同次数的智能体，并让迭代不同次数的智能体两两对弈，并绘制一个胜率矩阵，通过实验结果分析智能体的训练次数与胜率之间的关系
### 二、初始代码介绍
#### 1.全局设置与State类
- **功能：** 定义4x4棋盘，管理游戏状态
- **关键变量：** 
    - `BOARD_ROWS`, `BOARD_COLS`：棋盘尺寸(3x3)。
    - `data`：二维数组，存储棋盘状态（1: 玩家1，-1: 玩家2，0: 空）。
    - `hash_val`：唯一哈希值，用于快速识别状态。
- **核心方法：** 
  **`hash()`**：将棋盘状态转换为唯一整数，便于存储和比较。
```python
def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1     # 0, 1, -1 -> 1, 2, 0 因为哈希值不能为负数
        return self.hash_val
```
> &emsp;&emsp;这段代码使用三进制哈希值将不同的棋盘状态映射到唯一的一个整数，便于存储。
> - 哈希值唯一性表示
>   - 三进制性质：每个棋盘位置的权重为$3^{15-pos}$（例如第一个位置权重为$3^{15}$，最后一个为$3^{0}$）
>   - 数学证明：假设两个不同棋盘状态在某个位置的值不同，则它们的哈希值至少相差$3^{k}$（k是差异位置），因此哈希值必然不同。
> - 棋盘状态的数学表示
>   - 每个棋盘位置有3种状态：空（0）、玩家1（1）、玩家2（-1）。
>   - 3x3棋盘共有9个位置，总共有$3^{9}$=19,683种可能的棋盘状态。
>   - 4x4棋盘共有16个位置，总共有$3^{16}$=43,046,721种可能的棋盘状态。
> - 三进制编码
>   - **基数选择**：使用 3 作为进制基数（每个位置有3种状态）。
>   - **值转换**：将每个位置的值从 [-1, 0, 1] 映射到 [0, 1, 2]：self.hash_val * 3 + i **+ 1**
> - 哈希生成过程
>   - 将棋盘数据展开为9个连续的值（例如按行遍历）。
>   - **累乘累加**：依次将每个位置的值加入哈希计算：
>                   hash_val = hash_val * 3 + (current_value)


&emsp;**`is_end()`**：检查游戏是否结束（某行/列/对角线全为1/-1，或棋盘已满）。
```python
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)
        for result in results:
            if result == 4:
                self.winner = 1
                self.end = True
                return self.end
            if result == -4:
                self.winner = -1
                self.end = True
                return self.end
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end
        self.end = False
        return self.end
```
> &emsp;&emsp;在这段代码中通过遍历每一行和每一列以及两条对角线的所有元素的和来判断棋局是否结束。若和为3，玩家1胜利，若和为-3，玩家2胜利。当棋盘满足平局条件（棋盘上各个所有元素的绝对值的和为棋盘大小）仍未分出胜负时则判为平局。
#### 2.状态生成器（get_all_states）
- **功能：** 递归生成所有可能的棋盘状态。
- **实现：** 遍历每个空位，生成新状态并检查游戏结束条件。使用哈希值存储状态以避免重复。
```python
def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)
def get_all_states():
    time_start = time.time()
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    time_end = time.time()
    print('搜索所有状态所用时间为',time_end - time_start,'s')
    return all_states
```
> &emsp;&emsp;在入口函数`get_all_states()`中首先初始化空棋盘状态，然后将初始状态存入字典，最后调用递归函数get_all_states_impl展开所有可能状态。
> &emsp;&emsp;get_all_states_impl()函数的递归逻辑如下：
>  - 遍历棋盘：逐个检查每个位置是否为空。
>  - 生成新状态：在空位落子（current_symbol为当前玩家符号）。
>  - 哈希去重：通过哈希值判断是否已记录该状态。
>  - 终止判断：
>     - 若新状态已结束（胜利/平局），停止递归。
>     - 否则切换玩家符号（-current_symbol），继续递归生成后续状态。

#### 3.Judger类
- **功能：** 管理对弈流程，控制玩家交替行动。
- **关键方法：** 
    - `play()`：执行一局游戏，返回胜者（1/-1/0）。通过交替调用玩家的`act()`方法进行落子，并更新状态。
    - `alternate()`：生成玩家交替的迭代器。
```python
class Judger:     (具体代码省略)
    def __init__(self, player1, player2):
        ......
    def reset(self):
        ......
    def alternate(self):
        ......
    def play(self, print_state=False):
        ......
```
> - **初始化**：
>   - 创建交替生成器，重置玩家，设置初始棋盘。
> - **行动循环**：
>   - 通过`next(alternator)`获取当前玩家。
>   - 调用`player.act()`获取落子位置 (i,j) 和符号 symbol。
>   - 生成新状态的哈希值，从预生成的`all_states`字典中获取新状态对象。
> - **状态更新**：
>   - 更新双方玩家对当前状态的感知`（set_state）`。
>   - 若开启 `print_state`，实时打印棋盘变化。
> - **终止条件**：
>   - 当 `is_end=True` 时，返回胜者（1：玩家1胜，-1：玩家2胜，0：平局）。

#### 4.Player类(智能体)
- **功能：** 强化学习智能体，通过TD学习更新策略。
- **核心机制：**
    - **epsilon-greedy策略：** 以概率epsilon随机探索，否则选择价值最高的动作。
    - **TD学习：** 通过时间差分更新状态价值估计（estimations字典）。
- **关键方法：**
    - `act()：`选择动作，更新greedy标志以区分探索/利用。
    - `backup()：`根据TD误差反向更新状态价值。
- **训练与保存：** `save_policy()`和`load_policy()`用于保存/加载训练后的策略。
```python
class Player:   (具体代码省略)
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        ......
    def reset(self):
        ......
    def set_state(self, state):
        ......
    def set_symbol(self, symbol):
        ......
    # 使用TD时间差分的方法更新值函数
    def backup(self):
        ......
    # 基于当前状态选择一个动作
    def act(self):
        ......
    def save_policy(self):
        ......
    def load_policy(self):
        ......
```
> `Player`类实现了一个基于时间差分（TD）学习和ε-greedy策略的强化学习智能体，用于3x3井字棋对弈。以下是各部分的详细说明：
> ##### 状态价值初始化`set_symbol`
> - **逻辑**：
>   - 为所有预生成状态设置初始价值：
>       - **胜利状态**：价值1.0（最高奖励）。
>       - **平局状态**：价值0.5（中性奖励）。
>       - **失败状态**：价值0.0（惩罚）。
>       - **未终结状态**：价值0.5（鼓励探索）。
> - **意义**：引导智能体向高价值状态（胜利）移动。
> ##### 动作选择 `act`（ε-greedy策略）
> - **合法动作生成**：遍历棋盘所有空位，生成可能的动作列表。
> - **探索/利用决策**：
>       - **随机探索**：以epsilon概率随机选择动作（避免局部最优）。。
>       - **贪婪利用**：选择预测价值最高的动作。
> - **动作格式**：返回[i, j, symbol]，包含坐标和玩家符号。
> #####  价值更新 `backup`（TD学习）
> - **TD(0)更新：**
>   - **公式**：$Q(s_t)←Q(s_t)+\alpha[Q(s_{t+1})-Q(s_t)]$
>   - 仅当动作为贪婪选择时更新（self.greedy[i]为1）。
> - **反向传播**：从最后一个非终止状态向前更新，利用后续状态价值修正当前估计。
> ##### 训练与保存：`save_policy()`和`load_policy()`用于保存/加载训练后的策略。

#### 5.训练与对弈函数
- **train(epochs)**：训练两个智能体进行指定次数的对弈，每局结束后更新策略。定期输出胜率。
- **compete(turns)**：加载训练好的策略，让两个智能体对弈指定次数，统计胜率。
#### 6.人机交互（HumanPlayer类）
```python
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['1','2','3','4','q','w','e','r','a','s', 'd', 'f', 'z', 'x', 'c', 'v']
        self.state = None
    def reset(self):
        pass
    def set_state(self, state):
        self.state = state
    def set_symbol(self, symbol):
        self.symbol = symbol
    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol
```
- **功能**：允许人类玩家通过键盘输入落子位置（使用预定义键位映射）。
- **方法**：`act()`接收输入，转换为棋盘坐标并返回

### 三、4x4代码修改
#### 棋盘大小修改
##### 3x3
```python
BOARD_ROWS = 3
BOARD_COLS = 3
```
##### 4x4
```python
BOARD_ROWS = 4
BOARD_COLS = 4
```
#### `State`类中`is_end()`函数判断胜利条件修改
##### 3x3
```python
for result in results:
    if result == 3:
        self.winner = 1
        self.end = True
        return self.end
    if result == -3:
        self.winner = -1
        self.end = True
        return self.end
```
##### 4x4
```python
for result in results:
    if result == 4:
        self.winner = 1
        self.end = True
        return self.end
    if result == -4:
        self.winner = -1
        self.end = True
        return self.end
```
#### `HumanPlayer`类中初始化玩家输入按键修改
##### 3x3
```python
def __init__(self, **kwargs):
    self.symbol = None
    self.keys = [,'q','w','e','a','s', 'd', 'z', 'x', 'c']
    self.state = None
```
##### 4x4
```python
def __init__(self, **kwargs):
    self.symbol = None
    self.keys = ['1','2','3','4','q','w','e','r','a','s', 'd', 'f', 'z', 'x', 'c', 'v']
    self.state = None
```
#### 使用pickle库将遍历得到的所有状态保存以便后续直接调用
&emsp;&emsp;由于4x4的棋盘共有16个位置，总共有$3^{16}$=43,046,721种可能的棋盘状态，每次递归获取所有状态非常花费时间，所以这里使用pickle库将状态字典保存下来，以便后续直接调用。
```python
# 遍历所有状态并保存到all_states.pkl
'''
# all possible board configurations
all_states = get_all_states()

# 将所有的状态保存到外部文件中以便后续直接使用
with open('all_states.pkl', 'wb') as f:
    pickle.dump(all_states, f)
'''
# 直接读取all_states.pkl中保存的数据作为all_states
# 从文件中加载 all_states
with open('all_states.pkl', 'rb') as f:
    all_states = pickle.load(f)
```
### 四、实验结果展示
#### 4x4训练结果与玩家对战演示
> 以player1与player2的epsilon各为0.1训练10000轮并保存策略。
> 此时两者以epsilon各为0对战1000轮胜率如下：
> ![alt text](image.png)
> ##### 与玩家对战演示：
> ![alt text](image-1.png)
> ![alt text](image-2.png)
#### 训练迭代不同轮数的智能体并对弈
> 以player1与player2的epsilon各为0.1训练1000、10000、100000轮并分别保存策略。
> 让对战双方采取刚才训练轮数不同所得到的策略分别进行对战1000次并将胜率绘制成表格。

|Player1|Player2|Player1胜率|Player2胜率|
|---|---|---|---|
|$10^3$|$10^3$|0.56|0.29|
|$10^4$|$10^3$|0.74|0.19|
|$10^3$|$10^4$|0.66|0.15|
|$10^5$|$10^3$|0.72|0.16|
|$10^5$|$10^4$|0.79|0.15|
|$10^3$|$10^5$|0.5|0.4|
|$10^5$|$10^5$|0|0|

> &emsp;&emsp;通过观察以上结果可以发现，在训练次数较低时，先手玩家比后手玩家存在较大优势；训练次数较多的智能体比训练次数较少的智能体更为智能，但往往不能弥补先手所带来的优势。
> &emsp;&emsp;训练次数较多的智能体比训练次数较少的智能体更为智能的一个典型表现为：当对弈双方训练次数都为$10^3$时，两者胜率不为零而且先手玩家胜率更高，而当对弈双方训练次数都为$10^5$时，两者胜率都为0，这是由于随着智能程度提升，对局更加倾向于平局。
