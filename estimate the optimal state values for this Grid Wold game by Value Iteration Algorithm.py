########################################################################################################################
# Gridworld game:
# -------------
# | s | r | g |
# -------------
# | t | g | w |
# -------------
# | w | w | w |
# -------------
# Goal: estimate the optimal state values for this Grid Wold game by Value Iteration Algorithm
# MDP : Infinite horizon， gamma = 0.9
# State_space = {(0,0), (0,1), (0,2), (1,0), (1,1)}
# Action_space = { 0=left, 1=up, 2=right, 3=down }
# Reward = {r_s = -1, r_r = -1, r_g = 10, r_t = -10}, a function of arriving state
# Transition prob: P(s|g or t, a) = 1, for any action a
# prob = 0.7 and 0.1(if not the acton selected)
#############################################################################################################
import numpy as np
from numpy import random
from numpy import linalg
import matplotlib.pyplot as plt

board = np.array([['s','r','g'],['t','g','w'],['w','w','w']])

def state_list(board):
    state_lis = []
    board_r, board_c = board.shape
    for i in range(board_r):
        for j in range(board_c):
            if board[i,j] != 'w':
                state_lis.append((i,j))
    return state_lis
#print(state_list(board))

def next_state(board, cur_pos, real_action, s_position=(0,0)):
    board_r, board_c = board.shape
    cur_r, cur_c = cur_pos
    if board[cur_r, cur_c] == 'w':
        return 'current position is w'
    elif board[cur_r, cur_c] == 't' or board[cur_r, cur_c] == 'g':
        return s_position
    else:
        if real_action == 0:
            dr, dc = 0, -1
        elif real_action == 1:
            dr, dc = -1, 0
        elif real_action == 2:
            dr, dc = 0, 1
        else:
            dr, dc = 1, 0
        tem_r = cur_r + dr
        tem_c = cur_c + dc
        if (tem_r < 0) or (tem_r >= board_r) or (tem_c < 0) \
                or (tem_c >= board_c) or (board[tem_r, tem_c] == 'w'):
            return cur_pos
        else:
            return (tem_r, tem_c)
#print(next_state(board, (0,1), 0))

def tg_sub_dict(board, s_position=(0,0)):
    state_lis = state_list(board)
    tg_sub_dic = {}
    lis = []
    for s in state_lis:
        if s == s_position:
            lis.append(1)
        else:
            lis.append(0)
    for a in range(4):
        tg_sub_dic[a] = lis
    return tg_sub_dic
#print(tg_sub_dict(board))

def sr_sub_dict(board, cur_pos, pyes=0.7, pno=0.1):
    state_lis = state_list(board)
    sr_sub_dic = {}
    for a in range(4):
        lis = [0 * j for j in range(len(state_lis))]
        for i in range(4):
            next_pos = next_state(board, cur_pos, i)
            if i == a:
                lis[state_lis.index(next_pos)] += pyes
            elif i != a:
                lis[state_lis.index(next_pos)] += pno
        sr_sub_dic[a] = lis
    return sr_sub_dic
#print(sr_sub_dict(board, (0,0)))
#print(sr_sub_dict(board, (0,1)))

def prob_dict(board):
    state_lis = state_list(board)
    prob_dict = {}
    for s in state_lis:
        if board[s] == 't' or board[s] == 'g':
            prob_dict[s] = tg_sub_dict(board)
        elif board[s] == 's' or board[s] == 'r':
            prob_dict[s] = sr_sub_dict(board, s)
    return prob_dict
#print(prob_dict(board))

def init_values(board, lower, upper):
    state_lis = state_list(board)
    init = []
    for i in range(len(state_lis)):
        init.append(random.randint(lower,upper))
    init = np.array(init)
    return init
#print(init_values(board, -100, 100))

def reward(board):
    state_lis = state_list(board)
    r = []
    for s in state_lis:
        if board[s] == 's' or board[s] == 'r':
            r.append(-1)
        elif board[s] == 't':
            r.append(-10)
        elif board[s] == 'g':
            r.append(10)
    r=np.array(r)
    return r
#print(reward(board))


def value_iteration(board, gamma=0.9, epsilon=1e-4):
    state_lis = state_list(board)
    threshold = epsilon * (1 - gamma) / (2 * gamma)
    r = reward(board)
    prob_dic = prob_dict(board)
    # for this specific grid world, pi(right|s)=1 for all s is an optimal policy,
    # so, the true_optimal values were obtained by solving the linear system
    true_optimal = np.array([12.7656, 18.133, 10.489, 10.489, 10.489])
    old_values = init_values(board, -100, 100)
    values_list = []
    values_list.append(old_values)
    n = 0
    while linalg.norm(old_values-true_optimal, np.inf) > threshold:
        n += 1
        print('iteration ', n-1)
        print('diff norm = ', linalg.norm(old_values-true_optimal, np.inf))
        new_values = []
        for s in state_lis:
            tem_q = []
            for a in range(4):
                q = np.dot(np.array(prob_dic[s][a]), r) + \
                    gamma * np.dot(np.array(prob_dic[s][a]),old_values)
                tem_q.append(q)
            new_values.append(max(tem_q))
        old_values = new_values
        values_list.append(np.array(old_values))
    diff = []
    for i in range(len(values_list)):
        diff.append(linalg.norm(true_optimal - values_list[i], np.inf))
    return diff

diff = value_iteration(board, 0.9, 1e-2)
print(diff)

plt.plot(range(len(diff)),diff,'m--')
plt.xlabel("Iteration n")
plt.ylabel("Difference From Optimal Value")
plt.show()
