import pickle
import numpy as np
import torch
import torch.nn.functional as F
from DQN_f_sec import Environment, DQN
from data_preprocessing import load_data, remove_all_zero_rows, normalize_data
import argparse
import time
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For CPU and CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



seed = 42
set_seed(seed)

output_file = 'data/dataset_frequency_143_cut.csv'
X, y = load_data(output_file)
# model = TabPFNClassifier(N_ensemble_configurations=3, device='cuda')
model_filename = 'model/tabpfn_acc905_143cut.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)
model.device = 'cuda'
X = normalize_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train, overwrite_warning=True)
X = X[500:]
y = y[500:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(rf_model.evaluate(X_test,y_test))

env = Environment(model, X_train, y_train)
env2 = Environment(model, X_test, y_test)
parser = argparse.ArgumentParser(description='DQN')

parser.add_argument('--buffer_size', type=int, default=10000, help='replay buffer size')
parser.add_argument('--minimal_size', type=int, default=512, help='mini buffer size limit')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon greedy')

args = parser.parse_args()

agent = DQN(state_dim=X_train.shape[1], action_dim=X_train.shape[1], buffer_size=args.buffer_size,
            batch_size=args.batch_size, epsilon=args.epsilon, gamma=args.gamma, lr=args.lr)

episode = 0
tmp = np.zeros(10)  
best_percent = 0
end_time_best = 0
start_time0 = time.time()
agent.epsilon = 1.0  
epsilon_decay = 0.995  
min_epsilon = 0.01

best_eva = 0
best_eva_10 = 0
best_eva_11 = 0
best_eva_12 = 0
best_eva_13 = 0
best_eva_14 = 0
best_eva_15 = 0

while 1:
    num_test = 0
    print("{}#########################################".format(episode))
    episode += 1
    action_list = []
    for idx in range(X_train.shape[0]):
        state = env.reset(idx)
        zero_mask = (state == 0).astype(float)
        zero_mask = np.array(zero_mask, dtype=np.float32)
        # print(zero_mask.shape)
        label = y_train[idx]
        agent.reset_dynamic_mask()
        done = False
        # print(state)
        while not done:
            action = agent.choose_action(state)
            action_list.append(action)
            agent.dynamic_mask[action] = 0
            if len(action_list) == 71:
                next_states, rewards, done = env.step(action_list, label)

                for action, next_state, reward in zip(action_list, next_states, rewards):
                    reward = reward * 100
                    if reward:
                        agent.replay_buffer.push(state, action, reward, next_state, done, zero_mask)
                    agent.dynamic_mask[action] = 0
                    state = next_state
                action_list = []
            if len(agent.replay_buffer) > args.minimal_size:
                b_s, b_a, b_r, b_ns, b_d, b_m = agent.replay_buffer.sample(args.batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d, 'zero_masks': b_m}
                agent.update(transition_dict)
        if agent.epsilon > min_epsilon:
            agent.epsilon *= epsilon_decay
        else:
            agent.epsilon = min_epsilon
        if env.list:
            actions_str = ", ".join(str(action) for action in env.list)
            count_str = ", ".join(str(action) for action in env.count_num)
            accuracy_str = ", ".join(str(action) for action in env.accuracyx)
            print(
                f"train:\nindex={idx},   features : {actions_str}\nindex={idx},      count : {count_str}\nindex={idx},   accuracy : {accuracy_str}")
    def get_dqn_output(agent, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        out = agent.model(X_tensor)
        return out
    num_features = X_train.shape[1]  
    X_full_1 = np.ones(num_features)  
    out_train = get_dqn_output(agent, X_full_1)
    # out_train = get_dqn_output(agent, X_train[0])
    def get_key_features_by_q_values(out, top_k):
        out_cpu = out.cpu().detach().numpy()
        # key_features_t = np.argsort(out_cpu, axis=1)[:, ::-1]
        key_features_t = np.argsort(out_cpu)[::-1]
        key_features_top_k = key_features_t[:top_k]
        return key_features_top_k
    key_features = get_key_features_by_q_values(out_train, 15)


    action_list = key_features
    # length = len(action_list)
    print(f"key_features: {key_features}")
    sum_eva_10 = 0
    sum_eva_11 = 0
    sum_eva_12 = 0
    sum_eva_13 = 0
    sum_eva_14 = 0
    sum_eva_15 = 0


    
    
    for idx in range(X_test.shape[0]):
        state = env2.reset(idx)
        label = y_test[idx]
        next_state, reward, done = env2.step2(action_list, label)
        if env2.list:
            actions_str = ", ".join(str(action) for action in env2.list)
            count_str = ", ".join(str(action) for action in env2.count_num)
            accuracy_str = ", ".join(str(action) for action in env2.accuracyx)
            print(
                f"test:\nindex={idx},   features : {actions_str}\nindex={idx},   accuracy : {accuracy_str}")
            if len(env2.accuracyx) <= 11:
                sum_eva_10 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_11 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_12 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_13 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_14 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_15 += env2.accuracyx[0] - env2.accuracyx[-1]
            elif len(env2.accuracyx) <= 12:
                sum_eva_10 += env2.accuracyx[0] - env2.accuracyx[11]
                sum_eva_11 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_12 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_13 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_14 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_15 += env2.accuracyx[0] - env2.accuracyx[-1]
            elif len(env2.accuracyx) <= 13:
                sum_eva_10 += env2.accuracyx[0] - env2.accuracyx[11]
                sum_eva_11 += env2.accuracyx[0] - env2.accuracyx[12]
                sum_eva_12 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_13 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_14 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_15 += env2.accuracyx[0] - env2.accuracyx[-1]
            elif len(env2.accuracyx) <= 14:
                sum_eva_10 += env2.accuracyx[0] - env2.accuracyx[11]
                sum_eva_11 += env2.accuracyx[0] - env2.accuracyx[12]
                sum_eva_12 += env2.accuracyx[0] - env2.accuracyx[13]
                sum_eva_13 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_14 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_15 += env2.accuracyx[0] - env2.accuracyx[-1]
            elif len(env2.accuracyx) <= 15:
                sum_eva_10 += env2.accuracyx[0] - env2.accuracyx[11]
                sum_eva_11 += env2.accuracyx[0] - env2.accuracyx[12]
                sum_eva_12 += env2.accuracyx[0] - env2.accuracyx[13]
                sum_eva_13 += env2.accuracyx[0] - env2.accuracyx[14]
                sum_eva_14 += env2.accuracyx[0] - env2.accuracyx[-1]
                sum_eva_15 += env2.accuracyx[0] - env2.accuracyx[-1]
            elif len(env2.accuracyx) <= 16:
                sum_eva_10 += env2.accuracyx[0] - env2.accuracyx[11]
                sum_eva_11 += env2.accuracyx[0] - env2.accuracyx[12]
                sum_eva_12 += env2.accuracyx[0] - env2.accuracyx[13]
                sum_eva_13 += env2.accuracyx[0] - env2.accuracyx[14]
                sum_eva_14 += env2.accuracyx[0] - env2.accuracyx[15]
                sum_eva_15 += env2.accuracyx[0] - env2.accuracyx[-1]    
            
        state = next_state[-1]
        acc = model.predict_proba(state.reshape(1, -1))[:, label]
        if acc <= 0.5:
            num_test += 1
    num_percent = num_test / X_test.shape[0]

    eva_10 = sum_eva_10 / X_test.shape[0]
    eva_11 = sum_eva_11 / X_test.shape[0]
    eva_12 = sum_eva_12 / X_test.shape[0]
    eva_13 = sum_eva_13 / X_test.shape[0]
    eva_14 = sum_eva_14 / X_test.shape[0]
    eva_15 = sum_eva_15 / X_test.shape[0]


    if eva_10 > best_eva_10:
        best_eva_10 = eva_10
        agent.save_model("dqn_best_ab10.pth")
        print(f"best_eva_10: {best_eva_10}")
    
    if eva_11 > best_eva_11:
        best_eva_11 = eva_11
        agent.save_model("dqn_best_ab11.pth")
        print(f"best_eva_11: {best_eva_11}")
    
    if eva_12 > best_eva_12:
        best_eva_12 = eva_12
        agent.save_model("dqn_best_ab12.pth")
        print(f"best_eva_12: {best_eva_12}")

    if eva_13 > best_eva_13:
        best_eva_13 = eva_13
        agent.save_model("dqn_best_ab13.pth")
        print(f"best_eva_13: {best_eva_13}")
    
    if eva_14 > best_eva_14:
        best_eva_14 = eva_14
        agent.save_model("dqn_best_ab14.pth")
        print(f"best_eva_14: {best_eva_14}")

    if eva_15 > best_eva_15:
        best_eva_15 = eva_15
        agent.save_model("dqn_best_ab15.pth")
        print(f"best_eva_15: {best_eva_15}")
    
    if 0 <= num_percent < 0.1:
        if tmp[0] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time0 = time.time()
            tmp[0] = 1
            print(f"num_percent first appear between 0 ~ 0.1 : {end_time0 - start_time0} seconds")
            agent.save_model("dqn_model_f_0.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.1 <= num_percent < 0.2:
        if tmp[1] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time1 = time.time()
            tmp[1] = 1
            print(f"num_percent first appear between 0.1 ~ 0.2 : {end_time1 - start_time0} seconds")
            agent.save_model("dqn_model_f_1.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.2 <= num_percent < 0.3:
        if tmp[2] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time2 = time.time()
            tmp[2] = 1
            print(f"num_percent first appear between 0.2 ~ 0.3 : {end_time2 - start_time0} seconds")
            agent.save_model("dqn_model_f_2.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.3 <= num_percent < 0.4:
        if tmp[3] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time3 = time.time()
            tmp[3] = 1
            print(f"num_percent first appear between 0.3 ~ 0.4 : {end_time3 - start_time0} seconds")
            agent.save_model("dqn_model_f_3.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.4 <= num_percent < 0.5:
        if tmp[4] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time4 = time.time()
            tmp[4] = 1
            print(f"num_percent first appear between 0.4 ~ 0.5 : {end_time4 - start_time0} seconds")
            agent.save_model("dqn_model_f_4.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.5 <= num_percent < 0.6:
        if tmp[5] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time5 = time.time()
            tmp[5] = 1
            print(f"num_percent first appear between 0.5 ~ 0.6 : {end_time5 - start_time0} seconds")
            agent.save_model("dqn_model_f_5.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.6 <= num_percent < 0.7:
        if tmp[6] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time6 = time.time()
            tmp[6] = 1
            print(f"num_percent first appear between 0.6 ~ 0.7 : {end_time6 - start_time0} seconds")
            agent.save_model("dqn_model_f_6.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.7 <= num_percent < 0.8:
        if tmp[7] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time7 = time.time()
            tmp[7] = 1
            print(f"num_percent first appear between 0.7 ~ 0.8 : {end_time7 - start_time0} seconds")
            agent.save_model("dqn_model_f_7.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    elif 0.8 <= num_percent < 0.9:
        if tmp[8] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time8 = time.time()
            tmp[8] = 1
            print(f"num_percent first appear between 0.8 ~ 0.9 : {end_time8 - start_time0} seconds")
            agent.save_model("dqn_model_f_8.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
    else:
        if tmp[9] == 0:
            if best_percent < num_percent:
                best_percent = num_percent
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")
            end_time9 = time.time()
            tmp[9] = 1
            print(f"num_percent first appear between 0.9 ~ 1.0 : {end_time9 - start_time0} seconds")
            agent.save_model("dqn_model_f_9.pth")
        else:
            if num_percent > best_percent:
                best_percent = num_percent
                end_time_best = time.time()
                print(f"best_percent: {best_percent}")
                agent.save_model("dqn_model_f_best.pth")

    print(f"num_percent = {num_percent}")

    end_time_f = time.time()
    if end_time_f - start_time0 > 259200:
        break

print(f"best appear : {end_time_best - start_time0} seconds")
agent.save_model("dqn_model_f.pth")