##################迭代策略评估算法实现#########################
import numpy as np
import copy
#矩阵扩维和降维度
#print("P_pi",Pi[1,:],np.expand_dims(Pi[1,:],axis=0))
#print(np.dot(np.expand_dims(Pi[1,:],axis=0),P_ssa[:,1,:]).squeeze())
class Maze:
    def __init__(self):
        #初始化行为值函数
        self.qvalue = -10*np.ones((16,4))
        #初始化每个状态-动作对的次数
        self.C = 0*np.ones((16,4))
        self.states = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        self.actions = np.array([0,1,2,3])
        self.gamma = 1.0
        #初始化采样策略
        self.behaviour_Pi = 0.25 * np.ones((16, 4))
        #初始化目标策略
        self.target_Pi = np.zeros((16,4))
        for i in range(16):
            j = np.random.choice(self.actions, p=[0.25, 0.25, 0.25, 0.25])
            self.target_Pi[i,j]=1
        self.Greedy_Pi = np.zeros((16,4))
        self.cur_state = 1
        self.cur_action = 0
        self.old_policy=np.ones((16,4))
        #################1.状态转移概率P(s'|s,a)模型构建#################################
        self.P_ssa = np.zeros((4, 16, 16))
        # print(P_ssa)
        P1 = np.zeros((16, 16))
        P2 = np.zeros((16, 16))
        P3 = np.zeros((16, 16))
        P4 = np.zeros((16, 16))
        # 状态1处的转移
        P1[1, 2] = 1
        P2[1, 5] = 1
        P3[1, 0] = 1
        P4[1, 1] = 1
        # 状态2处的转移
        P1[2, 3] = 1
        P2[2, 6] = 1
        P3[2, 1] = 1
        P4[2, 2] = 1
        # 状态3处的转移
        P1[3, 3] = 1
        P2[3, 7] = 1
        P3[3, 2] = 1
        P4[3, 3] = 1
        # 状态4处的转移
        P1[4, 5] = 1
        P2[4, 8] = 1
        P3[4, 4] = 1
        P4[4, 0] = 1
        # 状态5处的转移
        P1[5, 6] = 1
        P2[5, 9] = 1
        P3[5, 4] = 1
        P4[5, 1] = 1
        # 状态6处的转移
        P1[6, 7] = 1
        P2[6, 10] = 1
        P3[6, 5] = 1
        P4[6, 2] = 1
        # 状态7处的转移
        P1[7, 7] = 1
        P2[7, 11] = 1
        P3[7, 6] = 1
        P4[7, 3] = 1
        # 状态8处的转移
        P1[8, 9] = 1
        P2[8, 12] = 1
        P3[8, 8] = 1
        P4[8, 4] = 1
        # 状态9处的转移
        P1[9, 10] = 1
        P2[9, 13] = 1
        P3[9, 8] = 1
        P4[9, 5] = 1
        # 状态10处的转移
        P1[10, 11] = 1
        P2[10, 14] = 1
        P3[10, 9] = 1
        P4[10, 6] = 1
        # 状态11处的转移
        P1[11, 11] = 1
        P2[11, 15] = 1
        P3[11, 10] = 1
        P4[11, 7] = 1
        # 状态12处的转移
        P1[12, 13] = 1
        P2[12, 12] = 1
        P3[12, 12] = 1
        P4[12, 8] = 1
        # 状态13处的转移
        P1[13, 14] = 1
        P2[13, 13] = 1
        P3[13, 12] = 1
        P4[13, 9] = 1
        # 状态14处的转移
        P1[14, 15] = 1
        P2[14, 14] = 1
        P3[14, 13] = 1
        P4[14, 10] = 1
        self.P_ssa[0, :, :] = P1
        self.P_ssa[1, :, :] = P2
        self.P_ssa[2, :, :] = P3
        self.P_ssa[3, :, :] = P4
        ###############2.回报模型构建########################
        self.r_sa = -np.ones((16, 4))
        self.r_sa[0, :] = 0
        self.r_sa[15, :] = 0
        self.epsilon = 0.5
        # self.r_sa[1,2]=1
    #重置环境函数
    def reset(self):
        # 初始化行为值函数
        self.qvalue = -10*np.zeros((16, 4))
        # 初始化每个状态-动作对的次数
        self.C = 0* np.ones((16, 4))
        self.alpha = 0.1
    #根据采样策略采样一个动作
    def sample_action(self,state):
        action = np.random.choice(self.actions,p=self.behaviour_Pi[state,:])
        return action
    #跟环境交互一步
    def step(self,action):
        flag = False
        trans_prob = self.P_ssa[action, self.cur_state]
        if(self.cur_state==0 or self.cur_state==15):
            next_state = self.cur_state
        else:
            next_state = np.random.choice(self.states,p=trans_prob)
        r_next = self.r_sa[self.cur_state,action]
        if next_state == 0 or next_state==15:
            flag = True
        #print(self.cur_state,action,next_state)
        return next_state,r_next,flag
    #############更新目标策略##########
    def update_target_policy(self):
        for i in range(16):
            self.target_Pi[i, :] = 0
            max_num = np.argmax(self.qvalue[i, :])
            self.target_Pi[i, max_num] = 1
    #############更新采样策略##########
    def update_behaviour_policy(self):
        for i in range(16):
            self.behaviour_Pi[i, :] = self.epsilon / 4
            max_num = np.argmax(self.qvalue[i, :])
            self.behaviour_Pi[i, max_num] = self.epsilon / 4 + (1 - self.epsilon)
    #############获得贪婪策略##########
    def get_greedy_policy(self):
        actions=['e','s','w','n']
        for i in range(16):
            self.Greedy_Pi[i, :] = 0
            max_num = np.argmax(self.qvalue[i, :])
            self.Greedy_Pi[i, max_num] = 1
            print(i,'->',actions[max_num])
        return self.Greedy_Pi
     #Qlearning强化学习算法
    def Qlearning(self):
        num = 0
        self.update_target_policy()
        self.update_behaviour_policy()
        delta =100
        q_former = copy.deepcopy(self.qvalue)
        while delta>0.01:
            self.cur_state=1
            self.epsilon = self.epsilon - 1 / 20000
            cur_action = self.sample_action(self.cur_state)
            flag=False
            num+=1
            #每采样100条轨迹，检查下值函数是否收敛
            if num%101 ==0:
                delta = np.linalg.norm(q_former-self.qvalue)
                q_former = copy.deepcopy(self.qvalue)
            while flag==False:
                #与环境交互一次
                next_state, reward, flag = self.step(cur_action)
                # 策略评估
                if flag== True:
                    self.qvalue[self.cur_state,cur_action]=reward
                    break
                else:
                    next_action = self.sample_action(next_state)
                    #在策略评估步不同,r+self.gamma*max(q[s',:])
                    td_target = reward+np.max(self.qvalue[next_state,:])
                    self.qvalue[self.cur_state,cur_action]=self.qvalue[self.cur_state,cur_action]+\
                    self.alpha*(td_target-self.qvalue[self.cur_state,cur_action])
                #策略改善
                self.update_target_policy()
                self.update_behaviour_policy()
                #环境往前推进一步
                self.cur_state = next_state
                cur_action = next_action
        print("num:", num)
def q_ana_evaluate(Pi,r_sa,P_ssa):
    P_pi = np.zeros((16, 16))
    C_pi = np.zeros((16, 1))
    for i in range(16):
        # 计算pi(a|s)*p(s'|s,a)
        P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
        # 计算pi(a|s)*r(s,a)
        C_pi[i, :] = np.dot(r_sa[i, :], Pi[i, :])
    ############解析法计算值函数######################
    M = np.eye(16) - P_pi
    I_M = np.linalg.inv(M)
    V = np.dot(I_M, C_pi)
    #计算行为值函数
    q_value = np.zeros((16, 4))
    for i in range(16):
        q_sa = np.zeros((1, 4))
        for j in range(4):
            Pi[i, :] = 0
            Pi[i, j] = 1
            P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
            vi = np.dot(r_sa[i, :], Pi[i, :]) + np.dot(P_pi[i, :], V.squeeze())
            q_sa[0, j] = vi
        q_value[i, :] = q_sa[0, :]
    return q_value



if __name__ == '__main__':
    ######实例化对象####
    maze=Maze()
    maze.reset()
    # maze.Off_MC_learning()
    # print(maze.get_greedy_policy())
    # print("估计值函数：\n",np.around(maze.qvalue,2))
    # q_real=q_ana_evaluate(maze.target_Pi,maze.r_sa,maze.P_ssa)
    # print("真实值函数：\n",np.around(q_real,2))
    # print("值函数差的范数：\n",np.linalg.norm(maze.qvalue-q_real))
    # print("访问频次：\n",np.around(maze.C,1))
    maze.Qlearning()
    print("最优epsilon-greedy策略为：\n",maze.target_Pi)
    print("评估值函数为：\n",maze.qvalue)
    q_real = q_ana_evaluate(maze.target_Pi, maze.r_sa, maze.P_ssa)
    print("真实值函数为：\n",q_real)
    print("最优贪婪策略为：\n",maze.get_greedy_policy())
