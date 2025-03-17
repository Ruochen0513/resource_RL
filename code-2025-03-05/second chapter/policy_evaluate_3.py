# TODO:使用策略迭代和值迭代算法求解上次挑选的Gym环境的最优策略

##################迭代策略评估算法实现#########################
import numpy as np
#矩阵扩维和降维度
#print("P_pi",Pi[1,:],np.expand_dims(Pi[1,:],axis=0))
#print(np.dot(np.expand_dims(Pi[1,:],axis=0),P_ssa[:,1,:]).squeeze())
##########利用解析法进行策略评估###########
def V_ana_evaluate(Pi,r_sa,P_ssa):
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
    return V
##########利用数值迭代法进行策略评估###########
def V_iter_evaluate(Pi,r_sa,P_ssa,V_init=np.zeros((16,1))):
    # 初始化当前值函数
    V_cur = V_init
    #计算C_pi和P_pi
    P_pi = np.zeros((16, 16))
    C_pi = np.zeros((16, 1))
    for i in range(16):
        # 计算pi(a|s)*p(s'|s,a)
        P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
        # 计算pi(a|s)*r(s,a)
        C_pi[i, :] = np.dot(r_sa[i, :], Pi[i, :])
    V_next = C_pi + np.dot(P_pi, V_cur)
    # 计算迭代一次的误差
    delta = np.linalg.norm(V_next - V_cur)
    num=0
    while delta > 1e-6:
        print("num=",num)
        print("V",V_cur)
        V_cur = V_next
        V_next = C_pi + np.dot(P_pi, V_cur)
        delta = np.linalg.norm(V_next - V_cur)
        num+=1
    print("num:",num)
    print("V_cur",V_cur)
    return V_cur
#############策略改进源代码##########         贪婪策略
def update_policy(r_sa,P_ssa,V):
    Pi_new = np.zeros((16, 4))
    Pi = np.zeros((16,4))
    # 计算C_pi和P_pi
    P_pi = np.zeros((16, 16))
    for i in range(16):
        q_sa = np.zeros((1, 4))
        for j in range(4):
            Pi[i, :] = 0
            Pi[i, j] = 1
            P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
            vi = np.dot(r_sa[i, :], Pi[i, :]) + np.dot(P_pi[i, :], V.squeeze())
            q_sa[0, j] = vi
        max_num = np.argmax(q_sa)
        Pi_new[i, max_num] = 1
    return Pi_new
###########策略迭代算法##########
def policy_iteration(Pi,r_sa,P_ssa):
    Pi_cur = Pi
    #策略评估
    V_cur = V_iter_evaluate(Pi_cur,r_sa,P_ssa)
    #V_cur = V_ana_evaluate(Pi_cur, r_sa, P_ssa)
    #策略改进
    Pi_new = update_policy(r_sa,P_ssa,V_cur)
    delta =  np.linalg.norm(Pi_new-Pi_cur)
    iter_num = 1
    while delta>1e-6:
        Pi_cur = Pi_new
        # 策略评估
        V_cur = V_iter_evaluate(Pi_cur, r_sa, P_ssa,V_cur)
        #V_cur = V_ana_evaluate(Pi_cur, r_sa, P_ssa)
        # 策略改进
        Pi_new = update_policy(r_sa, P_ssa, V_cur)
        delta = np.linalg.norm(Pi_new - Pi_cur)
        iter_num=iter_num+1
    return Pi_cur,iter_num


if __name__ == '__main__':
    #################1.状态转移概率P(s'|s,a)模型构建#################################
    P_ssa = np.zeros((4, 16, 16))
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
    P_ssa[0, :, :] = P1
    P_ssa[1, :, :] = P2
    P_ssa[2, :, :] = P3
    P_ssa[3, :, :] = P4
    #############当前策略模型##############
    Pi = 0.25 * np.ones((16, 4))
    ###############2.回报模型构建########################
    r_sa = -np.ones((16, 4))
    r_sa[0, :] = 0
    r_sa[15, :] = 0
    # V_cur = V_ana_evaluate(Pi,r_sa,P_ssa)
    # print(V_cur)
    # Pi_new = upadte_policy(Pi,r_sa,P_ssa,V_cur)
    # V_cur = V_ana_evaluate(Pi_new, r_sa, P_ssa)
    # print(V_cur)
    Pi_optim,iter_num = policy_iteration(Pi,r_sa,P_ssa)
    print(Pi_optim,iter_num)




