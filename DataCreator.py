import random
import sys
import numpy

from Blackjack import Blackjack

def datacoll(num_games):
        
    wins = 0
    pushes = 0
    loses = 0
    blackjack = Blackjack()
        
    Data_coll = []   #initialize the data set for my bayes method 
    for _ in range(num_games):
        finished, result, dealer_cards, player_cards = blackjack.play_game(Blackjack.ACTION["start"]) #I can name these output variables differently and create my output data differently
        hit_count = 0
        game_data = [Blackjack.get_card_value(player_cards),Blackjack.get_card_value([dealer_cards[0]]),result,hit_count,_+1]
        Data_coll.append(game_data)
        while not finished:
            action = random.choice([Blackjack.ACTION["hit"], Blackjack.ACTION["hold"]])
            finished, result, dealer_cards, player_cards = blackjack.play_game(action)
            if action == Blackjack.ACTION["hit"]:
                hit_count += 1
                Data_coll.append([Blackjack.get_card_value(player_cards),Blackjack.get_card_value([dealer_cards[0]]),result,hit_count,_+1])
            else:
                Data_coll.append([Blackjack.get_card_value(player_cards),Blackjack.get_card_value([dealer_cards[0]]),result,hit_count,_+1])
        
        wins += 1 if result == Blackjack.RESULT["win"] else 0
        pushes += 1 if result == Blackjack.RESULT["push"] else 0
        loses += 1 if result == Blackjack.RESULT["lose"] else 0
    return (Data_coll)

def dataprocc(data,game):       #breaks down the data into game sequence lists
    datasize = len(data)
    current_game = []
    for i in range(datasize):   #uses a for loop to break down the data lists into the actions of a game
        if game == data[i][4] :  #identifies game of the current item.
            current_game.append(data[i])   
    return(current_game)

def gameprocc(game_data):
    datasize = len(game_data)   #identifies the number of moves made in the game, it is actually one more than the number of moves
    outcome = game_data[datasize-1][2]       #i am only going to record data for the last move because each player move of hit or hold is independent of each other
    player_value = game_data[0][0]  #the starting card value of the player
    dealer_value = game_data[0][1]  #the starting dealer value of his shown card
    features = []
    if datasize == 2 :
        if  abs(player_value - game_data[datasize-1][0]) > 0 :
            hit = 1
        else:
            hit = 0
        features = [outcome, player_value,dealer_value,hit]  #[win(1)/lose(-1)/pus(0), player value, dealer value of shown card, hit/hold ]
    elif player_value == 21:
        features = [outcome, player_value,dealer_value,0]
    return(features)
    
if __name__ == '__main__':
    
    num_games = 10
    currentgame = 1    
    AllData = datacoll(num_games)
    Gamemove = []

    for i in range(num_games):
        cur_ga_mov = dataprocc(AllData,currentgame)
        Gamemove.append(gameprocc(cur_ga_mov))
        currentgame += 1
    Gamemove = [x for x in Gamemove if x != []]
    
    print(Gamemove)   
