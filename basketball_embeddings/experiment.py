#################### training experiments of basketball embedding models ######################
#

import tensorflow as tf
import numpy as np
from etl import preprocess_files, GameBatchGenerator
from model import GameEmbeddings, summation, aveg, concat
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

embedding_size = 50
steps = 2000
batch_size = 32
model_type = 'n'
agg_function = concat

# data, player_dict = preprocess_files(['/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games_boxscore.csv'],
#                        "/home/nether-nova/Documents/AIGaming/basketball_embeddings/data/all_games.csv")
data = pickle.load(open('data.pickle', "rb"))
player_dict = pickle.load(open('player_dict.pickle', "rb")) # [person_ids, emb_ids]

player_name_df = pd.read_csv('./data/player_ids')
player_name_df = player_name_df[['PERSON_ID', 'DISPLAY_LAST_COMMA_FIRST']]
player_name_dict = player_name_df.set_index('PERSON_ID').T.to_dict('list') # [person_id, name]
# reverse_player_name_dict = dict(zip(player_name_dict.values(), player_name_dict.keys()))

print player_name_dict

player_name_dict_new = dict()
for k,v in player_dict.iteritems():
    try:
        # [emb_id, name]
        player_name_dict_new[v] = player_name_dict[k]
    except KeyError:
        continue

data = np.array(data)
np.random.shuffle(data)

train_size = int(np.floor(0.7 * len(data)))
valid_size = int(np.floor(0.2 * len(data)))

train_data = data[:train_size]
valid_data = data[train_size : train_size + valid_size]
test_data = data[train_size + valid_size :]

print "Train: ", train_data.shape
print "Valid: ", valid_data.shape
print "Test: ", test_data.shape

train_loss = []
valid_loss = []

train_acc = []
valid_acc = []

def make_feed_dict(batch):
    team_a_batch = []
    team_b_batch = []
    score_a_batch = []
    score_b_batch = []
    for train_ex in batch:
        team_a_batch.append(np.array(train_ex[0]))
        team_b_batch.append(np.array(train_ex[1]))
        score_a_batch.append(train_ex[2])
        score_b_batch.append(train_ex[3])
    feed_dict = {
        model.team_a: team_a_batch,
        model.team_b: team_b_batch,
        model.score_team_a: score_a_batch,
        model.score_team_b: score_b_batch
    }
    return feed_dict, score_a_batch, score_b_batch


def plot_embeddings(embs, reverse_dictionary, top_k = None):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embs)
    df = pd.DataFrame(low_dim_embs[top_k, :], columns=['x1', 'x2'])
    sns.lmplot('x1', 'x2', data=df, scatter=True, fit_reg=False)

    for i in range(low_dim_embs.shape[0]):
        if i not in reverse_dictionary or i not in top_k:
            continue
        x, y = low_dim_embs[i, :]
        plt.annotate(reverse_dictionary[i],
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

gbg = GameBatchGenerator(data, batch_size)
valid_batch_generator = GameBatchGenerator(valid_data, len(valid_data))

model = GameEmbeddings(len(player_dict), embedding_size, agg_function, model_type)
model.build_model()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    avg_loss = 0
    for step in range(steps):
        batch = gbg.next()
        feed_dict, score_a_batch, score_b_batch = make_feed_dict(batch)
        loss, op, score_a, score_b = sess.run(model.updates(), feed_dict=feed_dict)
        avg_loss += loss
        if step % 50 == 0:
            print "Average loss train: %.2f" %(avg_loss / 50.0)
            train_loss.append(loss)
            avg_loss = 0
            team_a_wins = score_a > score_b
            team_a_actual_wins = np.array(score_a_batch) > np.array(score_b_batch)
            test = team_a_wins == team_a_actual_wins
            train_accurcay = np.sum(test) / (1.0*test.shape[1])
            print "Win accuracy train: %.2f" %(train_accurcay)
            train_acc.append(train_accurcay)

            # Validation
            valid_batch = valid_batch_generator.next()
            feed_dict, score_a_batch, score_b_batch = make_feed_dict(valid_batch)
            loss, score_a, score_b = sess.run([model.loss, model.pred_team_a, model.pred_team_b], feed_dict=feed_dict)
            valid_loss.append(loss)
            team_a_wins = score_a > score_b
            team_a_actual_wins = np.array(score_a_batch) > np.array(score_b_batch)
            test = team_a_wins == team_a_actual_wins
            valid_accuracy = np.sum(test) / (1.0 * test.shape[1])
            print "Win accuracy valid: %.2f " %(valid_accuracy)
            valid_acc.append(valid_accuracy)

    test_batch_generator = GameBatchGenerator(test_data, len(test_data))
    test_batch = test_batch_generator.next()
    feed_dict, score_a_batch, score_b_batch = make_feed_dict(test_batch)
    score_a, score_b = sess.run([model.pred_team_a, model.pred_team_b], feed_dict=feed_dict)
    team_a_wins = score_a > score_b
    team_a_actual_wins = np.array(score_a_batch) > np.array(score_b_batch)
    test = team_a_wins == team_a_actual_wins
    print "Win accuracy test: %.2f" %(np.sum(test) / (1.0 * test.shape[1]))

    x = range(len(train_acc))
    # plt.plot(x, train_acc, color='blue', label='train')
    # plt.plot(x, valid_acc, color='green', label='valid')
    #
    # plt.show()
    #
    # plt.plot(x[1:], train_loss[1:], color='blue', label='train')
    # plt.plot(x[1:], valid_loss[1:], color='green', label='valid')
    #
    # plt.show()

    gsw = np.array([player_dict[201939], player_dict[202691], player_dict[201142], player_dict[203110], player_dict[2585]])
    cle = np.array([player_dict[202681], player_dict[2747], player_dict[2544], player_dict[201567], player_dict[202684]])

    feed_dict = {
        model.team_a: gsw.reshape((1, 5)),
        model.team_b: cle.reshape((1, 5))
    }
    print "GSW @ CLE"
    print sess.run([model.pred_team_a, model.pred_team_b], feed_dict=feed_dict)

    cle = np.array([player_dict[202681], player_dict[2747], player_dict[2544], player_dict[201567], player_dict[165]])
    feed_dict = {
        model.team_a: gsw.reshape((1, 5)),
        model.team_b: cle.reshape((1, 5))
    }
    print "GSW @ CLE (Dream)"
    print sess.run([model.pred_team_a, model.pred_team_b], feed_dict=feed_dict)

    # top-k of Jordan Offensive
    W_off = sess.run([model.W_off])[0]
    sim = W_off[player_dict[893], :].dot(W_off.T)
    sim = sim * -1
    top_k = sim.argsort()[:10]

    plot_embeddings(W_off, player_name_dict_new, top_k)

    # top-k of Jordan Defensive
    W_deff = sess.run([model.W_deff])[0]
    sim = W_deff[player_dict[893], :].dot(W_deff.T)
    sim = sim * -1
    top_k = sim.argsort()[:10]

    plot_embeddings(W_deff, player_name_dict_new, top_k)