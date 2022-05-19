import numpy as np
import pickle
import random
from numpy.linalg import norm

# board build up
room_row = 3
room_col = 3
room_num_inrow = 2
room_num_incol = 2
board = np.zeros((room_row*room_num_incol+2+room_num_incol-1, room_col*room_num_inrow+2+room_num_inrow-1))
r,c = board.shape
board[0,:] = -1
board[r-1, :] = -1
board[:,0] = -1
board[:, c-1] = -1
# 假定每个隔墙的第二个位置是door
door_pos=2
for i in range(1, room_num_inrow):
    board[:, (room_col+1)*i] = -1
    for j in range(room_num_incol):
        board[(room_row+1)*j+door_pos, (room_col+1)*i] = 0
for i in range(1, room_num_incol):
    board[(room_row+1)*i,:] = -1
    for j in range(room_num_inrow):
        board[(room_row+1)*i, (room_col+1)*j+door_pos] = 0
#print(board)

agent_pos = (1,1)
exit_pos = (r-2, c-2)
def printboard(board, agent_pos, exit_pos):
    r,c = board.shape
    board[agent_pos] = -3
    board[exit_pos] = -2
    board_pri = []
    for i in range(r):
        for j in range(c):
            if board[i,j] == -1:
                board_pri.append('W')
            elif board[i,j] == -2:
                board_pri.append('D')
            elif board[i,j] == -3:
                board_pri.append('*')
            else:
                board_pri.append(' ')
    for i in range(r):
        print(board_pri[c*i: c*(i+1)])
printboard(board, agent_pos, exit_pos)

#这个要改
def true_value(reward, dis_factor):
    true_power = [12,12,     11, 11, 13,   10, 12,         8, 8,   9,7,9,   8,8,
                  13,11,11,  12,10,10,12,  11,9,11,11,     8,10,   9,7,7,9,  8,8,6,8,   9,7,7,
                  12,10,     11,11,9,11,   10,10,          8,6,    7,7,5,7,  8,6,
                  10,8,      6,4,
                  8,8,       9,7,7,9,      6,8,            4,4,    5,3,3,5,  2,4,
                  9,7,9,     8,6,8,8,      7,5,7,7,        4,6,    5,3,3,5,  4,2,2,4,   3,1,3,
                  8,8,       7,7,9,        6,8,            4,2,    3,1,3]
    true_values = []
    for i in range(len(true_power)):
        true_values.append(reward*(dis_factor**true_power[i]))
    true_values_array = np.array(true_values)
    return true_values_array
#print(true_value(10, 0.8))
#print(len(true_value(10,0.8)))

#value_iteration 用下一个state代替action，而这里用1，2，3，4代表上右下左
def valid_action(board, agent_pos):
    valid_action = []
    ar,ac = agent_pos
    if board[ar-1, ac] != -1:
        valid_action.append(1)
    if board[ar, ac+1] != -1:
        valid_action.append(2)
    if board[ar+1, ac] != -1:
        valid_action.append(3)
    if board[ar, ac-1] != -1:
        valid_action.append(4)
    random.shuffle(valid_action)
    return valid_action
#print(valid_action(board, agent_pos))

# up=1, right=2, down=3, left=4
def init_state_action_value_dic(board):
    s_a_value_dic = {}
    r,c = board.shape
    for i in range(r):
        for j in range(c):
            if board[i,j] != -1 and board[i,j] != -2:
                if board[i-1, j] != -1:
                    s_a_value_dic[(i,j,1)] = 0
                if board[i, j+1] != -1:
                    s_a_value_dic[(i,j,2)] = 0
                if board[i+1, j] != -1:
                    s_a_value_dic[(i,j,3)] = 0
                if board[i, j-1] != -1:
                    s_a_value_dic[(i,j,4)] = 0
    return s_a_value_dic
#print(init_state_action_value_dic(board))
#print(len(init_state_action_value_dic(board)))

s_a_value_dic = init_state_action_value_dic(board)

#注意，这里action的返回值3维，的（current, agent_pos, action）
def choose_action(board, agent_pos, s_a_value_dic, eps=0.3):
    ar, ac = agent_pos
    valid_actions = valid_action(board, agent_pos)
    rn = random.uniform(0, 1)
    if rn <= eps:
        action_ind = np.random.choice(len(valid_actions))
        action = (ar, ac, valid_actions[action_ind])
    else:
        max_val = -1000000001
        for i in range(len(valid_actions)):
            if s_a_value_dic[(ar, ac, valid_actions[i])] > max_val:
                max_val = s_a_value_dic[(ar, ac, valid_actions[i])]
                action = (ar, ac, valid_actions[i])
    return action
#print(choose_action(board, agent_pos, s_a_value_dic, 0.3))

def train(board, training_num, dis_factor = 0.8, eps=0.3, reward=10):
    r,c = board.shape
    exit_pos = (r-2, c-2)
    s_a_value_dic = init_state_action_value_dic(board)
    true_values = true_value(reward, dis_factor)
    last_values = np.zeros(len(s_a_value_dic))
    for i in range(training_num):
        for j in range(r-2,-1,-1):
            for k in range(c-2, -1, -1):
                if board[j,k] != -1 and board[j,k] != -2:
                    agent_pos = (j,k)
                    for h in valid_action(board,agent_pos):
                        if h == 1:
                            tem_pos = (j-1, k)
                        elif h == 2:
                            tem_pos = (j, k+1)
                        elif h == 3:
                            tem_pos = (j+1, k)
                        else:
                            tem_pos = (j, k-1)
                        steps = 1

                        while tem_pos != exit_pos:
                            steps += 1
                            action = choose_action(board, tem_pos, s_a_value_dic, eps)
                            x,y = tem_pos
                            if action[-1] == 1:
                                tem_pos = (x-1, y)
                            elif action[-1] == 2:
                                tem_pos = (x, y+1)
                            elif action[-1] == 3:
                                tem_pos = (x+1, y)
                            else:
                                tem_pos = (x, y-1)
                        s_a_value_dic[(j,k,h)] += 1/(i+1)*((dis_factor**steps)*reward-s_a_value_dic[(j,k,h)])
        current_values = np.array(list(s_a_value_dic.values()))
        diff = current_values - last_values
        true_diff = current_values - true_values
        # 由于true_value 是按照3*3的room2行2列写的code，所以与true_value的 l2—norm 对比也只能对比这种
        print('iteration= ', i, ' l2= ', norm(diff,2), ' true norm= ', norm(true_diff,2))
        # 其他排列只对比上下代iteration
        #print('iteration= ', i, ' l2= ', norm(diff,2))
        last_values = current_values
    fw_1 = open('gridword_Q_learning_' + str(training_num), 'wb')
    pickle.dump(s_a_value_dic, fw_1)
    fw_1.close()
training_num = 200
train(board, training_num, 0.1, 0.01, 1)

def test(board, agent_pos, train_num):
    r,c = board.shape
    exit_pos = (r-2, c-2)
    #下面这三行是在测试训练成果
    fr = open('gridword_Q_learning_'+str(train_num), 'rb')
    state_value_dic = pickle.load(fr)
    fr.close()
    #下面这个是在测试dict=0是的策略，不是最优policy
    #state_value_dic = init_state_value_dic(board, exit_pos)
    n = 0
    print('Initial board:')
    printboard(board, agent_pos, exit_pos)
    while agent_pos != exit_pos:
        n += 1
        valid_actions = valid_action(board, agent_pos)
        max_val = -100000001
        for i in range(len(valid_actions)):
            if state_value_dic[valid_actions[i]] > max_val:
                max_val = state_value_dic[valid_actions[i]]
                action = valid_actions[i]
        agent_pos = action
        printboard(board, agent_pos, exit_pos)
    print('win after '+ str(n)+' steps!')
#test(board, agent_pos, 2000)
