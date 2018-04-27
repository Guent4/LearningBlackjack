import numpy as np

from Blackjack import Blackjack


class BlackjackEnv():
    def __init__(self):
        self._seed = None
        self._blackjack = None

        self.reset()

    def seed(self, seed=None):
        self._seed = seed

    def reset(self):
        self._blackjack = Blackjack(self._seed)

    def deal(self):
        _, _, dealer_cards, player_cards = self._blackjack.play_game(Blackjack.ACTION["start"])
        return dealer_cards, player_cards

    def step(self, action):
        assert action in Blackjack.ACTION.values()

        action = Blackjack.ACTION["hold"] if action == 0 else Blackjack.ACTION["hit"]

        finished, result, dealer_cards, player_cards = self._blackjack.play_game(action)
        reward = BlackjackEnv._calculate_reward(finished, result, player_cards)

        return [dealer_cards, player_cards], reward, finished, [result]

    @staticmethod
    def _calculate_reward(finished, result, player_cards):
        if finished and result == Blackjack.RESULT["lose"]:
            # Player lost
            return -100
        elif finished and result == Blackjack.RESULT["win"]:
            # Player won
            return 100
        elif finished and result == Blackjack.RESULT["push"]:
            # Push
            return -20
        elif not finished:
            # Still in the middle of a hand
            return 0


def Q_learning_with_epsilon_greedy():
    def cards_to_state(dealer_cards, player_cards, only_one_dealer_card=True):
        dealer_cards_interested_in = [dealer_cards[0]] if only_one_dealer_card else dealer_cards
        player_value = Blackjack.get_card_value(player_cards)
        dealer_value = Blackjack.get_card_value(dealer_cards_interested_in)
        assert player_value >= 4
        assert dealer_value >= 2
        return player_value - 1, dealer_value - 1

    # Number of iterations
    num_episodes_learning = 100000
    num_episodes_evaluating = 10000

    # Use the custom BlackjackEnv
    env = BlackjackEnv()

    # Set up some hyperparameters
    q_table = np.zeros((21, 11, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999

    env.reset()
    for i in range(num_episodes_learning):
        if i % 100 == 0:
            print("Learning episode {} of {}".format(i + 1, num_episodes_learning))

        # Update hyperparameters
        eps *= decay_factor

        # Deal next hand
        s = cards_to_state(*env.deal())
        done = False

        while not done:
            # select the action with highest cumulative reward
            if np.random.random() < eps or np.sum(q_table[s[0], s[1], :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s[0], s[1], :])

            # Take the step and then update the q_table
            new_s_cards, r, done, _ = env.step(a)
            if done:
                new_s = cards_to_state(*new_s_cards, only_one_dealer_card=False)  # This is just to see if we won or not
                q_table[s[0], s[1], a] += r + lr * (- q_table[s[0], s[1], a])
            else:
                new_s = cards_to_state(*new_s_cards)
                q_table[s[0], s[1], a] += r + lr * (y * np.max(q_table[new_s[0], new_s[1], :]) - q_table[s[0], s[1], a])
            s = new_s

    # Save the q_table to a csv
    with open("hold.csv", "wb") as hold:
        np.savetxt(hold, q_table[:, :, 0])
    with open("hit.csv", "wb") as hit:
        np.savetxt(hit, q_table[:, :, 1])

    # Evaluate how good this q_table is
    env_eval = BlackjackEnv()
    env_eval.seed("eval")
    env_eval.reset()

    wins = 0
    pushes = 0
    loses = 0

    for i in range(num_episodes_evaluating):
        if i % 100 == 0:
            print("Evaluation episode {} of {}".format(i + 1, num_episodes_evaluating))

        # Update hyperparameters
        eps *= decay_factor

        # Deal next hand
        s = cards_to_state(*env.deal())
        done = False

        while not done:
            # select the action with highest cumulative reward
            if np.random.random() < eps or np.sum(q_table[s[0], s[1], :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s[0], s[1], :])

            # Take the step based on the learned Q-table
            new_s_cards, r, done, extra = env.step(a)
            result = extra[0]
            if done:
                wins += 1 if result == Blackjack.RESULT["win"] else 0
                pushes += 1 if result == Blackjack.RESULT["push"] else 0
                loses += 1 if result == Blackjack.RESULT["lose"] else 0
            else:
                s = cards_to_state(*new_s_cards)

    return wins, pushes, loses


def usingKeras():
    import keras.layers

    # Convert given state as a tuple of dealer_cards and player_cards into a one-hot matrix
    def cards_to_state(dealer_cards, player_cards):
        s = np.zeros([1, 22])
        s[0, min(21, Blackjack.get_card_value(player_cards) - 1)] = 1
        return s


    # Keras model for the weights
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=(1, 22)))
    model.add(keras.layers.Dense(22 * 2, activation='sigmoid'))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Gym model that we will use to simulate stuff
    env = BlackjackEnv()

    # Number of iterations
    num_episodes = 10000

    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_sum_list = []
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(cards_to_state(*s)))
                print(a)
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(model.predict(cards_to_state(*new_s)))
            target_vec = model.predict(cards_to_state(*s))[0]
            target_vec[a] = target
            model.fit(cards_to_state(*s), target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        if r_sum == 45:
            print(s)
        r_sum_list.append(r_sum)

    with open("test.csv", "w") as file:
        for r_sum in r_sum_list:
            file.write("{}\n".format(r_sum))

    # plt.plot(r_sum_list)
    # plt.show()


print(Q_learning_with_epsilon_greedy())
