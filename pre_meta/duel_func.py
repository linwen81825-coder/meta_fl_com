import numpy as np

def update(Q_selection, r_dict, S):
    '''update the selection of each client'''
    r_sorted = sorted(r_dict.items(), key=lambda d: d[1], reverse=True)
    for i in range(S):
        for j in range(i+1, S):
            Q_selection[r_sorted[i][0],r_sorted[j][0]] += 1


def get_client_prob(Q_prob,K):
    client_prob = np.ones(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                client_prob[i] = client_prob[i] * Q_prob [i, j]
    return client_prob


def cal_Q_prob(Q_selection, K):
    '''cal the probability of wins Q_prob[i,j] means the counts of i wins j'''
    Q_prob = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i != j:
                Q_prob[i,j] = Q_selection[i,j] / (Q_selection[i,j] + Q_selection[j,i])
    return Q_prob

def cal_w_conf(w,t,K,alpha):
    '''cal the ucb and lcb value of i wins j'''
    w_ucb = np.zeros((K,K))
    w_lcb = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i != j:
                if (w[i,j] + w[j,i]) == 0:
                    w_ucb[i,j] == 2
                    w_lcb[i,j] == 0
                else:
                    w_ucb[i,j] = w[i,j] / (w[i,j] + w[j,i]) + np.sqrt(alpha*np.log(t) / (w[i,j] + w[j,i]))
                    w_lcb[i,j] = w[i,j] / (w[i,j] + w[j,i]) - np.sqrt(alpha*np.log(t) / (w[i,j] + w[j,i]))
            else:
                w_ucb[i,j] = 1/2
                w_lcb[i,j] = 1/2
    return w_ucb, w_lcb


def cal_w_ts(w, K):
    '''cal the Thompson sampling value by w of i wins j'''
    w_ts = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i < j:
                w_ts[i,j] = np.random.beta(w[i,j]+1, w[j,i]+1)
                w_ts[j,i] = 1 - w_ts[i,j]
    return w_ts


def cal_lw_num(w_ucb,w_lcb,K):
    '''cal the # of ucb lose, lcb wins'''
    lose_num_ucb = np.zeros(K)
    win_num_lcb = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if w_ucb[i,j] < 1/2:
                lose_num_ucb[i] += 1
            if w_lcb[i,j] > 1/2:
                win_num_lcb[i] += 1
    return lose_num_ucb, win_num_lcb


# def cal_sd_set(l_u, w_l, K, S):
#     '''cal the selection set and discard set'''
#     selected_set = []
#     discard_set = []
#     for i in range(K):
#         larger = 0
#         smaller = 0
#         for j in range(K):
#             if (K - l_u[j]) < w_l[i]:
#                 larger += 1
#             if (K - w_l[j]) < l_u[i]:
#                 smaller += 1
#         if larger > K - S:
#             selected_set.append(i)
#         if smaller > S:
#             discard_set.append(i)
#     return selected_set, discard_set


def cal_sd_set(l_u, w_l, K, S):
    selected_set = np.array([], dtype=int)
    discard_set = np.array([],dtype=int)
    remain_set = np.arange(K,dtype=int)
    for i in range(K):
        larger = 0
        smaller = 0
        for j in range(K):
            if (K - l_u[j]) < w_l[i]:
                larger += 1
            if (K - w_l[j]) < l_u[i]:
                smaller += 1
        if larger > K - S:
            selected_set = np.append(selected_set, i)
        if smaller > S:
            discard_set = np.append(discard_set, i)
    if selected_set.size != 0:
        for i in selected_set:
            d_idx = np.where(remain_set==i)
            remain_set = np.delete(remain_set, d_idx)
    if discard_set.size != 0:
        for i in discard_set:
            d_idx = np.where(remain_set==i)
            remain_set = np.delete(remain_set,d_idx)
    return selected_set, discard_set, remain_set


def ts_rank(selected_set, discard_set, remain_set, w, K):
    # d_set = np.append(selected_set, discard_set)
    # w_remain = np.copy(w)
    # w_sort = sorted(d_set, reverse=True)

    # w_remain = np.delete(w, d_set, axis=0)
    # w_remain = np.delete(w_remain, d_set, axis=1)

    w_ts = cal_w_ts(w, K)

    win_num_ts = {}
    for i in remain_set:
        num_win = 0
        for j in remain_set:
            if w_ts[i, j] > 1 / 2:
                num_win += 1
        win_num_ts[i] = num_win

    return win_num_ts


def ra_sort(n_selected, ra_dict):
    '''select the rank order of the ra value'''
    ra_sorted = sorted(ra_dict.items(), key=lambda d: d[1], reverse=True)
    rank_order = np.zeros(n_selected,dtype=int)
    for i in range(n_selected):
        rank_order[i] = ra_sorted[i][0]
    return rank_order

def get_client_idx(w, t, num_clients, num_selected, confident_alpha):
    # cal ucb, lcb of p
    w_ucb, w_lcb = cal_w_conf(w, t, num_clients, confident_alpha)
    # obtain g, h
    num_l_ucb, num_w_lcb = cal_lw_num(w_ucb, w_lcb, num_clients)
    # s_set: reserved set, d_set: discard set, r_set; remain set, need to be sampled by Thompson Sampling (TS)
    s_set, d_set, r_set = cal_sd_set(num_l_ucb, num_w_lcb, num_clients, num_selected)
    # rank remain set by TS
    w_remain_ts = ts_rank(s_set, d_set, r_set, w, num_clients)
    # set B
    s_ts_idx = ra_sort(num_selected - s_set.size, w_remain_ts)
    # obtain the final selected set
    final_s_set = np.append(s_set, s_ts_idx)
    return final_s_set, s_set, d_set