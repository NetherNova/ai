################ Game / player data etl #################
# Unique ids for players
# batch data consisting of games with scoring outcomes of both teams


import pandas as pd
import numpy as np
import pickle

# GAME_ID,TEAM_ID,TEAM_ABBREVIATION,TEAM_CITY,PLAYER_ID,PLAYER_NAME,START_POSITION,
# COMMENT,MIN,OFF_RATING,DEF_RATING,NET_RATING,AST_PCT,AST_TOV,AST_RATIO,OREB_PCT,
# DREB_PCT,REB_PCT,TM_TOV_PCT,EFG_PCT,TS_PCT,USG_PCT,PACE,PIE

def unique_player_ids(df):
    player_ids = df['PLAYER_ID'].unique()
    new_ids = range(len(player_ids))
    player_dict = dict(zip(player_ids, new_ids))
    return player_dict


def preprocess_files(box_score_files, all_games_file):
    data = []
    # load game scores
    game_df = pd.read_csv(all_games_file)
    df_list = []
    for f in box_score_files:
        print "processing file %s" %(f)
        df = pd.read_csv(f,
                         names=["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_ID", "PLAYER_NAME",
                                "START_POSITION", "COMMENT", "MIN", "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT",
                                "AST_TOV", "AST_RATIO", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "EFG_PCT",
                                "TS_PCT", "USG_PCT", "PACE", "PIE"])
        df_list.append(df)
    df = pd.concat(df_list)
    # TODO: concatenate dfs
    player_dict = unique_player_ids(df)
    print "Num players: %d " % len(player_dict)
    all_game_ids = df['GAME_ID'].unique()
    print "Num games: %d " %(len(all_game_ids))
    df['MIN'] = df['MIN'].str.split(':').str[0]
    df['MIN'] = df['MIN'].convert_objects(convert_numeric=True)
    for num, game_id in enumerate(all_game_ids):
        print "%d out of %d games" %(num+1, len(all_game_ids))
        try:
            game_score_df = game_df[game_df['Game_ID'] == game_id]
        except:
            print "skipping invalid game id"
            continue

        sub_df = df[df['GAME_ID'] == game_id]
        teams = sub_df['TEAM_ID'].unique()
        if len(teams) != 2:
            print "skipping invalid game"
            continue
        team_a, team_b = teams[0], teams[1]
        team_a_df = sub_df[sub_df['TEAM_ID'] == team_a]
        team_b_df = sub_df[sub_df['TEAM_ID'] == team_b]
        team_a_score = game_score_df[game_score_df['Team_ID'] == team_a]['PTS'].values[0]
        team_b_score = game_score_df[game_score_df['Team_ID'] == team_b]['PTS'].values[0]

        # TODO: include second unit of players
        team_a_guards = team_a_df[team_a_df['START_POSITION'] == 'G']
        team_a_forwards = team_a_df[team_a_df['START_POSITION'] == 'F']
        team_a_center = team_a_df[team_a_df['START_POSITION'] == 'C']
        team_a_list = [player_dict[ind] for ind in team_a_guards['PLAYER_ID'].values] + \
                      [player_dict[ind] for ind in team_a_forwards['PLAYER_ID'].values] + \
                      [player_dict[ind] for ind in team_a_center['PLAYER_ID'].values]

        team_b_guards = team_b_df[team_b_df['START_POSITION'] == 'G']
        team_b_forwards = team_b_df[team_b_df['START_POSITION'] == 'F']
        team_b_center = team_b_df[team_b_df['START_POSITION'] == 'C']
        team_b_list = [player_dict[ind] for ind in team_b_guards['PLAYER_ID'].values] + \
                      [player_dict[ind] for ind in team_b_forwards['PLAYER_ID'].values] + \
                      [player_dict[ind] for ind in team_b_center['PLAYER_ID'].values]

        if (len(team_a_list) != 5):
            print "skipping"
            continue
        if (len(team_b_list) != 5):
            print "skipping"
            continue
        data.append([team_a_list, team_b_list, team_a_score, team_b_score])
    return data, player_dict


class GameBatchGenerator(object):
    def __init__(self, data, batch_size):
        self.data = data
        np.random.shuffle(self.data)
        self.batch_size = batch_size
        self.data_index = 0

    def next(self):
        if self.data_index + self.batch_size >= len(self.data):
            self.data_index = 0

        result = self.data[self.data_index : self.data_index + self.batch_size]
        self.data_index += 1
        return result


if __name__ == '__main__':
    data, player_dict = preprocess_files(['/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games_boxscore.csv',
                                          '/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games_boxscore_v2.csv',
                                          '/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games_boxscore_v3.csv',
                                          '/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games_boxscore_v4.csv'],
                     "/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games.csv")
    pickle.dump(data, open("data.pickle", "wb"))
    pickle.dump(player_dict, open("player_dict.pickle", "wb"))