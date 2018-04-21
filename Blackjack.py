import collections
import random

import sys


class Blackjack(object):
    NUM_DECKS = 2
    PERCENT_REMAINING_BEFORE_SHUFFLING = 0.25
    ACTION = {
        "start": 0,
        "hold": 1,
        "hit": 2
    }
    RESULT = {
        "lose": -1,
        "push": 0,
        "win": 1
    }

    def __init__(self):
        random.seed("blackjack")
        self.Card = collections.namedtuple("Card", ["suit", "number"])
        self.unused = []
        self.inuse = []
        self.used = []
        self.create_deck()

        self.dealer_cards = None
        self.player_cards = None

    def create_deck(self):
        self.unused = []
        for _ in range(Blackjack.NUM_DECKS):
            for suit in range(4):
                for number in range(1, 14):
                    self.unused.append(self.Card(suit, number))

        return self.unused

    def shuffle_if_needed(self):
        if len(self.unused) < Blackjack.PERCENT_REMAINING_BEFORE_SHUFFLING * (13 * 4 * Blackjack.NUM_DECKS):
            self.unused.extend(self.used)
            self.used = []
            random.shuffle(self.unused)

    def draw_card(self):
        self.shuffle_if_needed()
        card = self.unused.pop(0)
        self.inuse.append(card)

        return card

    def _discard_card(self, card):
        self.inuse.remove(card)
        self.used.append(card)

    @staticmethod
    def get_card_value(cards):
        num_aces = 0
        value = 0
        for card in cards:
            if card.number == 1:        # Ace
                num_aces += 1
                value += 11
            else:
                value += min(10, card.number)     # Jack, Queen, King count as 10

        while num_aces > 0 and value > 21:
            num_aces -= 1
            value -= 10

        return value

    def play_game(self, action):
        assert action in Blackjack.ACTION.values(), "Action {} is not valid"
        assert len(self.unused) + len(self.inuse) + len(self.used) == 13 * 4 * Blackjack.NUM_DECKS, len(self.unused) + len(self.inuse) + len(self.used)

        if action == Blackjack.ACTION["start"]:
            self.dealer_cards = [self.draw_card(), self.draw_card()]  # 0 is hidden and 1 is revealed
            self.player_cards = [self.draw_card(), self.draw_card()]
            if Blackjack.get_card_value(self.dealer_cards) == 21 and Blackjack.get_card_value(self.player_cards) == 21:
                finished = True
                result = Blackjack.RESULT["push"]
            elif Blackjack.get_card_value(self.dealer_cards) == 21:
                finished = True
                result = Blackjack.RESULT["lose"]
            elif Blackjack.get_card_value(self.player_cards) == 21:
                finished = True
                result = Blackjack.RESULT["win"]
            else:
                finished = False
                result = Blackjack.RESULT["lose"]
        elif action == Blackjack.ACTION["hold"]:
            while Blackjack.get_card_value(self.dealer_cards) < 17:
                self.dealer_cards.append(self.draw_card())
            if Blackjack.get_card_value(self.dealer_cards) > 21:
                finished = True
                result = Blackjack.RESULT["win"]
            elif Blackjack.get_card_value(self.dealer_cards) == Blackjack.get_card_value(self.player_cards):
                finished = True
                result = Blackjack.RESULT["push"]
            elif Blackjack.get_card_value(self.dealer_cards) > Blackjack.get_card_value(self.player_cards):
                finished = True
                result = Blackjack.RESULT["lose"]
            else:
                finished = True
                result = Blackjack.RESULT["win"]
        elif action == Blackjack.ACTION["hit"]:
            self.player_cards.append(self.draw_card())
            value = Blackjack.get_card_value(self.player_cards)
            if value > 21:
                finished = True
                result = Blackjack.RESULT["lose"]
            else:
                finished = False
                result = Blackjack.RESULT["lose"]
        else:
            raise ValueError("Invalid action: {}".format(action))

        dealer_cards = [x for x in self.dealer_cards]
        player_cards = [x for x in self.player_cards]
        if finished:
            for card in self.dealer_cards:
                self._discard_card(card)
            for card in self.player_cards:
                self._discard_card(card)

        return finished, result, dealer_cards, player_cards

