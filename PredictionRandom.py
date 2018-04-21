import random
import sys

from Blackjack import Blackjack

num_games = int(float(sys.argv[1]))

wins = 0
pushes = 0
loses = 0

blackjack = Blackjack()
for _ in range(num_games):
    finished, result, dealer_cards, player_cards = blackjack.play_game(Blackjack.ACTION["start"])
    while not finished:
        action = random.choice([Blackjack.ACTION["hit"], Blackjack.ACTION["hold"]])
        finished, result, dealer_cards, player_cards = blackjack.play_game(action)

    wins += 1 if result == Blackjack.RESULT["win"] else 0
    pushes += 1 if result == Blackjack.RESULT["push"] else 0
    loses += 1 if result == Blackjack.RESULT["lose"] else 0

print("wins:\t{}".format(wins))
print("pushes\t{}".format(pushes))
print("loses:\t{}".format(loses))