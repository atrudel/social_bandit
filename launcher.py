from game import Game
from partner import Partner
from player import Player
from strategies import AlternatingStrategy
from data import ConstantTrajectory


if __name__ == '__main__':
    n_turns = 80

    # Todo: dataset
    player = Player(AlternatingStrategy())
    partner1 = Partner(ConstantTrajectory(n_turns, 20))
    partner2 = Partner(ConstantTrajectory(n_turns, 50))

    game = Game(player, partner1, partner2)
    game.play_n_turns(n_turns)
    game.visualize()