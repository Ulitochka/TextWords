import os
from collections import Counter


data_path = '/home/m.domrachev/Data/TextWorld_train/'
files = [(os.path.join(data_path + file), file.split('.json')[0].replace('tw-cooking-', '').split('-')[0])
         for file in os.listdir(data_path) if file.endswith('.json')]
games_types = Counter([el[-1] for el in files])

predict_log_path = '/home/m.domrachev/repos/TextWorld/TextWord_starting_kit/approaches/lstm_pg/eda/predict_logs/log.txt'
with open(predict_log_path, 'r') as file:
    log = [el.split('\t') for el in file.readlines()]
    zero_games = [el[-1].replace('\n', '') for el in log if el[0] == ' 0 / 1000:']
    print(zero_games)

