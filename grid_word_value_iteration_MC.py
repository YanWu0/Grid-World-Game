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

def true_value(reward, dis_factor):
    true_power = [12,11,10,8,7,8,11,10,9,8,7,6,7,10,9,10,6,5,6,8,4,8,7,6,4,3,2,7,6,5,4,3,2,1,8,7,6,2,1,0]
    true_values = []
    for i in range(len(true_power)):
        true_values.append(reward*(dis_factor**true_power[i]))
    true_values_array = np.array(true_values)
    return true_values_array
#print(true_value(10, 0.8))

def valid_action(board, agent_pos):
    valid_action = []
    ar,ac = agent_pos
    if board[ar-1, ac] != -1:
        valid_action.append((ar-1, ac))
    if board[ar+1, ac] != -1:
        valid_action.append((ar+1, ac))
    if board[ar, ac-1] != -1:
        valid_action.append((ar, ac-1))
    if board[ar, ac+1] != -1:
        valid_action.append((ar, ac+1))
    random.shuffle(valid_action)
    return valid_action
#print(valid_action(board, agent_pos))

def init_state_value_dic(board, exit_pos):
    state_value_dic = {}
    r,c = board.shape
    for i in range(r):
        for j in range(c):
            if board[i,j] != -1:
                state_value_dic[(i,j)] = 0
    state_value_dic[exit_pos] = 10
    return state_value_dic
#print(init_state_value_dic(board, exit_pos))

#state_value_dic = init_state_value_dic(board)

def choose_action(board, agent_pos, state_value_dic, eps=0.3):
    valid_actions = valid_action(board, agent_pos)
    rn = random.uniform(0, 1)
    if rn <= eps:
        action_ind = np.random.choice(len(valid_actions))
        action = valid_actions[action_ind]
    else:
        max_val = -1000000001
        for i in range(len(valid_actions)):
            if state_value_dic[valid_actions[i]] > max_val:
                max_val = state_value_dic[valid_actions[i]]
                action = valid_actions[i]
    return action
#print(choose_action(board, agent_pos, state_value_dic, 0.3))

def train(board, training_num, dis_factor = 0.8, eps=0.3, reward=10):
    r,c = board.shape
    exit_pos = (r-2, c-2)
    state_value_dic = init_state_value_dic(board, exit_pos)
    true_values = true_value(reward, dis_factor)
    last_values = np.zeros(len(state_value_dic))
    for i in range(training_num):
        for j in range(r-2,-1,-1):
            for k in range(c-2, -1, -1):
                if board[j,k] != -1:
                    agent_pos = (j,k)
                    steps = 0
                    while agent_pos != exit_pos:
                        steps += 1
                        action = choose_action(board, agent_pos,state_value_dic, eps)
                        agent_pos = action
                    state_value_dic[(j,k)] += (1/(i+1))*((dis_factor**steps)*reward-state_value_dic[(j,k)])
        current_values = np.array(list(state_value_dic.values()))
        diff = current_values - last_values
        #true_diff = current_values - true_values
        print('iteration= ', i, ' l2= ', norm(diff,2))
        #print(' true norm= ', norm(true_diff,2))
        last_values = current_values
    fw_1 = open('gridword_' + str(training_num), 'wb')
    pickle.dump(state_value_dic, fw_1)
    fw_1.close()

training_num = 200
#train(board, training_num, 0.4, 0.05, 10)

def test(board, agent_pos, train_num):
    r,c = board.shape
    exit_pos = (r-2, c-2)
    #下面这三行是在测试训练成果
    fr = open('gridword_'+str(train_num), 'rb')
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
test(board, agent_pos, 200)






