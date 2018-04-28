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
        _, _, dealer_cards, player_cards, revealed_cards = self._blackjack.play_game(Blackjack.ACTION["start"])
        return dealer_cards, player_cards, revealed_cards

    def step(self, action):
        assert action in Blackjack.ACTION.values()

        action = Blackjack.ACTION["hold"] if action == 0 else Blackjack.ACTION["hit"]

        finished, result, dealer_cards, player_cards, revealed_cards = self._blackjack.play_game(action)
        reward = BlackjackEnv._calculate_reward(finished, result, player_cards)

        return [dealer_cards, player_cards, revealed_cards], reward, finished, [result]

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
            return -25
        elif not finished:
            # Still in the middle of a hand
            return 0


def Q_learning_with_epsilon_greedy():
    def cards_to_state(dealer_cards, player_cards, revealed_cards, only_one_dealer_card=True):
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
    eps = 0.3
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
                q_table[s[0], s[1], a] += lr * (r - q_table[s[0], s[1], a])
            else:
                new_s = cards_to_state(*new_s_cards)
                q_table[s[0], s[1], a] += lr * (r + y * np.max(q_table[new_s[0], new_s[1], :]) - q_table[s[0], s[1], a])
            s = new_s

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

        # Deal next hand
        s = cards_to_state(*env_eval.deal())
        done = False

        while not done:
            # select the action with highest cumulative reward
            if np.sum(q_table[s[0], s[1], :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s[0], s[1], :])

            # Take the step based on the learned Q-table
            new_s_cards, r, done, extra = env_eval.step(a)
            result = extra[0]
            if done:
                wins += 1 if result == Blackjack.RESULT["win"] else 0
                pushes += 1 if result == Blackjack.RESULT["push"] else 0
                loses += 1 if result == Blackjack.RESULT["lose"] else 0
            else:
                s = cards_to_state(*new_s_cards)

    return wins, pushes, loses


def learning_with_keras():
    import keras.layers

    def count_cards(cards):
        counts = [0 for _ in range(13)]
        for card in cards:
            counts[card.number - 1] += 1
        return counts

    # Convert given state as a tuple of dealer_cards and player_cards into a one-hot matrix
    def cards_to_state(dealer_cards, player_cards, revealed_cards, only_one_dealer_card=True):
        dealer_cards_interested_in = [dealer_cards[0]] if only_one_dealer_card else dealer_cards
        counts = count_cards(revealed_cards)

        s = np.zeros([1, 15])
        s[0, 0] = Blackjack.get_card_value(player_cards)
        s[0, 1] = Blackjack.get_card_value(dealer_cards_interested_in)
        for number in range(13):
            s[0, 2 + number] = counts[number]

        return s


    # Keras model for the weights
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(batch_input_shape=(1, 15)))
    model.add(keras.layers.Dense(15 * 2, activation='sigmoid'))
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Gym model that we will use to simulate stuff
    env = BlackjackEnv()

    # Set up some hyperparameters
    y = 0.95
    eps = 0.5
    decay_factor = 0.999

    # Number of iterations
    num_episodes_learning = 1000000
    num_episodes_evaluating = 10000

    env.reset()
    for i in range(num_episodes_learning):
        if i % 1000 == 0:
            print("Learning episode {} of {}".format(i + 1, num_episodes_learning))

        # Update hyperparameters
        eps *= decay_factor

        # Deal next hand
        s = cards_to_state(*env.deal())
        done = False

        while not done:
            # select the action with highest cumulative reward
            predicted_for_s = model.predict(s)
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(predicted_for_s)

            # Take the step and then update the q_table
            new_s_cards, r, done, _ = env.step(a)
            if done:
                new_s = None
                target = r
                target_vec = predicted_for_s[0]
                target_vec[a] = target
            else:
                new_s = cards_to_state(*new_s_cards)
                target = r + y * np.max(model.predict(new_s))
            target_vec = predicted_for_s[0]
            target_vec[a] = target
            model.fit(s, target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s

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
        s = cards_to_state(*env_eval.deal())
        done = False

        while not done:
            a = np.argmax(model.predict(s))

            # Take the step based on the learned Q-table
            new_s_cards, r, done, extra = env_eval.step(a)
            result = extra[0]
            if done:
                wins += 1 if result == Blackjack.RESULT["win"] else 0
                pushes += 1 if result == Blackjack.RESULT["push"] else 0
                loses += 1 if result == Blackjack.RESULT["lose"] else 0
            else:
                s = cards_to_state(*new_s_cards)

    return wins, pushes, loses

# print(Q_learning_with_epsilon_greedy())
print(learning_with_keras())