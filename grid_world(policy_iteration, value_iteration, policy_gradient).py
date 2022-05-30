########################################################################################################################
# implement the following gridworld game by policy_iteration, value_iteration, and policy_gradient
# ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
# ['W', '*', ' ', ' ', 'W', ' ', ' ', ' ', 'W']
# ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W']
# ['W', ' ', ' ', ' ', 'W', ' ', ' ', ' ', 'W']
# ['W', 'W', ' ', 'W', 'W', 'W', ' ', 'W', 'W']
# ['W', ' ', ' ', ' ', 'W', ' ', ' ', ' ', 'W']
# ['W', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'W']
# ['W', ' ', ' ', ' ', 'W', ' ', ' ', 'D', 'W']
# ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']
########################################################################################################################

########################################################################################################################
# Task 1: Iterative Policy Evaluation
# general formula for iterative policy evaluation, r=-1, gamma = 0, (undiscounted, finite, MDP)
# v(1,1) = p1 * ( -1 +v(1,1)) + p2 * (-1 + v(1,2)) + p3 * (-1 + v(2,1)) + p4 * (-1 + v(1,1))
#      = -1 + p1 * v(1,1) + p2 * v(1,2) + p3 * v(2,1) + p4 * v(1,1), where p1+p2+p3+p4 = 1 are prob
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

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
#printboard(board, agent_pos, exit_pos)


def next_state_collection(board):
    board_r, board_c = board.shape
    next_state_coll = {}
    for r in range(board_r):
        for c in range(board_c):
            if board[r,c] != -1 and [r,c] != [board_r-2,board_c-2]:
                current_next_state = []
                for i in range(4):
                    if i == 0:
                        tem_state = [r - 1, c]
                    elif i == 1:
                        tem_state = [r, c + 1]
                    elif i == 2:
                        tem_state = [r + 1, c]
                    else:
                        tem_state = [r, c - 1]
                    tem_r, tem_c = tem_state
                    if board[tem_r, tem_c] != -1:
                        current_next_state.append((tem_r,tem_c))
                    else:
                        current_next_state.append((r,c))
                next_state_coll[(r,c)] = current_next_state
    return next_state_coll
#print(next_state_collection(board))

next_state_coll = next_state_collection(board)
random_prob = [1/4, 1/4, 1/4, 1/4]
init_policy = {k:random_prob for k in next_state_coll.keys()}
init_V = {k:0 for k in next_state_coll.keys()}
init_V[(r-2,c-2)] = 0 # board[-1,-1]=Door, no policy, but it must has state value = 0， 所以value这里要加一项

# iterative policy evaluation
def iterative_policy_evluation(init_policy, init_V, board, theta):
    row_num, col_num = board.shape
    next_state_coll = next_state_collection(board)
    V = init_V
    policy = init_policy
    delta_list = []
    delta = theta
    while delta >= theta:
        delta = 0
        for r in range(row_num):
            for c in range(col_num):
                if board[r, c] != -1 and [r, c] != [row_num - 2, col_num - 2]:
                    sum_v = 0
                    for i in range(4):
                        #print(V[next_state_coll[(r,c)][i]])
                        #print(policy[next_state_coll[(r,c)][i]])
                        sum_v += V[next_state_coll[(r,c)][i]] * policy[(r,c)][i]
                    new_v = -1 + sum_v
                    delta = max(delta, abs(new_v - V[r, c]))
                    V[r, c] = new_v
        delta_list.append(delta)
    return delta_list, V

theta = 1e-8
#delta_list , V = iterative_policy_evluation(init_policy, init_V, board, theta)
#for k in V.keys():
#    V[k] = round(V[k], 0)
#print(V)
#n = len(delta_list)

#plt.plot(range(n),delta_list,'m--')
#plt.xlabel("iter_num")
#plt.ylabel("delta")
#plt.title('Iterative Policy Evaluation')
#plt.show()

########################################################################################################################
# Task 2: Policy Iteration
# general formula for policy improvement, r=-1, gamma = 0, (undiscounted, finite, MDP)
# pi(s) = argmax_a Sum_{s',r} p(s',r|s,a)[r+gamma*v(s')]
#      = argmax_a -1 + v(s')
########################################################################################################################

def policy_iteration(init_policy, init_V, board, theta):
    row_num, col_num = board.shape
    next_state_coll = next_state_collection(board)
    old_V = init_V
    old_policy = init_policy
    print(str(0)+' iter policy is:')
    print(old_policy)
    n = 0
    while True:
        n += 1
        old_V = iterative_policy_evluation(old_policy, old_V, board, theta)[1]
        # change old_V.values() into integers
        for k in old_V.keys():
            old_V[k] = round(old_V[k], 2)
        new_policy = {}
        for r in range(row_num):
            for c in range(col_num):
                if board[r,c] != -1 and [r,c] != [row_num - 2, col_num - 2]:
                    v_list = []
                    for i in range(4):
                        x,y = next_state_coll[(r,c)][i]
                        v_list.append(-1 + old_V[(x,y)])
                    v_max = max(v_list)
                    num_max = v_list.count(v_max)
                    p_list = []
                    for i in range(4):
                        if v_list[i] == v_max:
                            p_list.append(1/num_max)
                        else:
                            p_list.append(0)
                    new_policy[(r,c)] = p_list
        print(str(n)+' iter policy is:')
        print(new_policy)
        if new_policy == old_policy:
            break
        else:
            old_policy = new_policy
    return n,new_policy

#policy_iteration(init_policy, init_V, board, theta)


########################################################################################################################
# Task 3: Value Iteration
# general formula for policy improvement, r=-1, gamma = 0, (undiscounted, finite, MDP)
# formula simplification
# v(s) = max_a Sum_{s',r} p(s',r|s,a)[r + gamma* v(s')]
#      = max_a -1 + v(s')
########################################################################################################################

def true_value():
    true_values = np.array([[0, 0,   0,   0,   0,  0,  0,  0,  0],
                            [0, -12, -11, -10, 0,  -8, -7, -8, 0],
                            [0, -11, -10, -9,  -8, -7, -6, -7, 0],
                            [0, -10, -9,  -10, 0,  -6, -5, -6, 0],
                            [0, 0,   -8,  0,   0,   0, -4,  0, 0],
                            [0, -8,  -7,  -6,  0,  -4, -3, -2, 0],
                            [0, -7,  -6,  -5,  -4, -3, -2, -1, 0],
                            [0, -8,  -7,  -6,  0,  -2, -1, 0,  0],
                            [0, 0,   0,   0,   0,  0,  0,  0,  0]])
    return true_values
#print(true_value())

def value_iteration(init_V, board, theta):
    row_num, col_num = board.shape
    V = init_V
    next_state_coll = next_state_collection(board)
    true_values = true_value()
    delta_true_list = []
    delta_list = []
    delta_true = theta
    delta = theta
    n = 0
    while delta >= theta:
        n += 1
        print(str(n)+ ' iter')
        delta = 0
        delta_true = 0
        for r in range(row_num):
            for c in range(col_num):
                if board[r,c] != -1 and [r,c] != [row_num - 2, col_num - 2]:
                    v_list = []
                    for i in range(4):
                        x,y = next_state_coll[(r,c)][i]
                        v_list.append(-1 + V[(x,y)])
                    delta = max(delta, abs(V[(r,c)] - max(v_list)))
                    #print('delta:', delta)
                    V[(r,c)] = max(v_list)
                    delta_true = max(delta_true, abs(V[(r,c)] - true_values[(r,c)]))
        delta_list.append(delta)
        #print('delta_list:', delta_list)
        delta_true_list.append(delta_true)
        #print('delta_true_list:', delta_true_list)
        print(str(n)+'_iter:',V)
    # output is a deterministic policy
    optimal_policy = {}
    for r in range(row_num):
        for c in range(col_num):
            if board[r,c] != -1 and [r,c] != [row_num - 2, col_num - 2]:
                v_list = []
                for i in range(4):
                    x, y = next_state_coll[(r,c)][i]
                    v_list.append(V[(x,y)])
                action = v_list.index(max(v_list))
                p = []
                for j in range(4):
                    if j == action:
                        p.append(1.0)
                    else:
                        p.append(0.0)
                optimal_policy[(r,c)] = p
    return V, optimal_policy, delta_list, delta_true_list

# delta_list = difference between State_Value of n-th iteration and State_Value of n+1-th iteration
# init_V 如果都是0的话很快就找到了optimal value and policy
# 所以这里设 init_value 是random policy 对应的 state values
init_V = iterative_policy_evluation(init_policy, init_V, board, theta)[1]
optimal_V, optimal_policy, delta_list, delta_true_list = value_iteration(init_V, board, theta)
print(optimal_V)
print(optimal_policy)
n = len(delta_list)
print(n)

plt.plot(range(n),delta_list,'m--', label = 'Delta')
plt.plot(range(n),delta_true_list, 'r', label = 'Delta_true')
plt.xlabel("iter_num")
plt.ylabel("delta")
plt.legend(loc="upper right")
plt.show()
