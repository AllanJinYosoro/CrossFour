from assets.connectfour import *
import N0 as player1
import N29 as Rainbow
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def Test_10_each(Model_method):

    wins = []

    for _ in range(10):
        env = ConnectFourEnv(rows=6, cols=7, render_mode= None)
        # play_match(env, player1.policy1, player1.policy2)
        score = play_match(env, player1.policy1, Model_method)
        # play_match(env, "human", player1.policy2)

        wins.append(int(score == 2))

    for _ in range(10):
        env = ConnectFourEnv(rows=6, cols=7, render_mode= None)
        # play_match(env, player1.policy1, player1.policy2)
        score = play_match(env, Model_method, player1.policy1)
        # play_match(env, "human", player1.policy2)

        wins.append(int(score == 1))

    return np.mean(wins)

if __name__ == '__main__':
    win_rate = Test_10_each(Rainbow.policy1)
    print(win_rate)