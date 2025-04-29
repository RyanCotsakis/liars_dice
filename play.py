import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
import pickle

N_DICE = (1, 1) # tuple (number of dice P1, number of dice P2)
USE_PHYSICAL_DICE = False

class Die():
    faces = ["L", "2", "3", "4", "5", "6"] # Llamas first

    def __init__(self):
        self.roll()
    
    # roll the die n times and return the result as a list
    def roll(self, n=1):
        result = np.random.choice(Die.faces, n)
        self.result = result[-1]
        return result


class Player():
    action_space = np.concat(
        [[face * i for face in Die.faces] for i in range(1,np.sum(N_DICE) + 1)] +
        [["L" * i for i in range(np.sum(N_DICE) + 1, 2 * np.sum(N_DICE) + 1)]] +
        # [["D", "C"]] # doubt or call
        [["D"]] # just doubt
    )
    marked_for_removal = []
    for i, el in enumerate(action_space):
        if el[0] == "L":
            if len(el) % 2 == 1:
                marked_for_removal.append(i)
            else:
                action_space[i] = el[:len(el)//2]
    action_space = np.delete(action_space, marked_for_removal)

    def faceoff(player_0, player_1):
        player_0.opponent = player_1
        player_1.opponent = player_0
        if player_0.player_id == 0:
            player_0.act(grandfather_node)
        elif player_1.player_id == 0:
            player_1.act(grandfather_node)

    def __init__(self, player_id, human=False, physical_dice=False):
        if not human:
            physical_dice = False
        self.human = human
        self.opponent = None # corrected by Player.faceoff
        self.player_id = player_id
        self.die = Die()
        if not physical_dice:
            self.private = self.die.roll(N_DICE[player_id])
            if human:
                print("You rolled " + str(self.private))
        else:
            self.private = None
    
    def act(self, node):
        if self.human:
            self.act_manual(node)
        else:
            self.act_automatic(node)

    def act_manual(self, node):
        assert self.human
        def get_winner():
            if self.opponent.private is not None and self.private is not None:
                row = Node.roll2index(self.private)
                col = Node.roll2index(self.opponent.private)
                winner = node.winner[row, col].detach().numpy()
                if winner == self.player_id:
                    print("[Computer] You win.")
                else:
                    print("[Computer] I win!")

        if not self.opponent.human:
            if node.last_action is not None and node.last_action not in ("C", "D"):
                print("[Computer] I bet " + node.last_action + ".")
            elif node.last_action is None:
                print("[Computer] You start!")
            elif node.last_action == "D":
                print("[Computer] I doubt that. I had " + str(self.opponent.private))
                get_winner()
                return
            else:
                print("[Computer] I call. I had " + str(self.opponent.private))
                get_winner()
                return
        
        possible_actions = [action.item() for action in node.children]
        print("Possible actions: " + str(possible_actions) + " (without quotations)")
        action = input(">> ")
        print(action)
        assert action in possible_actions

        if self.private is None:
            possible_private = [str(Node.index2roll(i, N_DICE[self.player_id])) for i in range(len(Die.faces) ** N_DICE[self.player_id])]
            move_matrix = F.softmax(node.logits, dim=-1)
            df = pd.DataFrame(move_matrix.detach().numpy(), index=possible_private, columns=possible_actions)
        else:
            move_matrix = F.softmax(node.logits[Node.roll2index(self.private),:], dim=-1)
            df = pd.Series(move_matrix.detach().numpy(), index=possible_actions)
        print("The optimal moves were:")
        print(df)
        print()

        next_node = node.children[action]
        self.opponent.act(next_node)
        
    def act_automatic(self, node):
        assert not self.human
        if node.is_leaf and self.opponent.human:
            print("[Computer] I had " + str(self.private))
            if self.opponent.private is not None:
                row = Node.roll2index(self.private)
                col = Node.roll2index(self.opponent.private)
                winner = node.winner[row, col].detach().numpy()
                if winner == self.player_id:
                    print("[Computer] I win!")
                else:
                    print("[Computer] You win.")
            return
        move_matrix = F.softmax(node.logits[Node.roll2index(self.private),:], dim=-1).detach().numpy()
        next_node = np.random.choice([child for child in node.children.values()], p=move_matrix)
        self.opponent.act(next_node)


class Node():
    MAX_DEPTH = 4
    n_nodes = 0
    leaves = []
    deep_nodes = [] # these are not leaves

    def roll2index(roll):
        index = 0
        for i, face in enumerate(roll):
            d = Die.faces.index(face)
            index += (len(Die.faces) ** i) * d
        return index

    def index2roll(index, n_dice):
        roll = []
        for i in range(n_dice):
            roll.append(Die.faces[index % len(Die.faces)])
            index //= len(Die.faces)
        return roll

    def __init__(self, parent, last_action=None):
        self.parent = parent
        if parent is None:
            self.depth = 0
            self.player = 0 # start with player zero
            self.probability = torch.ones([len(Die.faces) ** n for n in N_DICE]) # certain to hit this node
            Node.player_logits = [[], []]
            Node.n_nodes = 0
        else:
            self.depth = self.parent.depth + 1
            self.player = (self.parent.player + 1) % 2
            self.probability = torch.zeros([len(Die.faces) ** n for n in N_DICE])
            if self.player:
                self.probability = self.probability.t()
        Node.n_nodes += 1
        self.last_action = last_action
        self.is_leaf = last_action == "D" or last_action == "C"
        self.children = {} # empty dictionary. Keys are actions and values are Nodes.
        if self.is_leaf:
            self.logits = None
            Node.leaves.append(self)
            self.compute_winner_matrix()
        else:
            self.winner = None
            if last_action is None:
                starting_index = 0
            else:
                starting_index = np.where(Player.action_space == last_action)[0][0] + 1
            possible_actions = Player.action_space[starting_index:]
            if last_action is None:
                possible_actions = np.delete(possible_actions,
                                             np.where([x[0] in ("L", "D", "C") for x in possible_actions])[0])
            n_actions = len(possible_actions)
            n_private = len(Die.faces) ** N_DICE[self.player]

            self.logits = torch.ones(n_private, n_actions, requires_grad=True)
            Node.player_logits[self.player].append(self.logits)
            
            if self.depth < Node.MAX_DEPTH:
                for action in possible_actions:
                    self.children[action] = Node(parent=self, last_action=action)
            else:
                previous_node = self
                for i in range(Node.MAX_DEPTH - 3):
                    previous_node = previous_node.parent
                would_start_with = previous_node.last_action
                if (would_start_with[0] == "L") or ((Node.MAX_DEPTH - self.player) % 2):
                    for action in possible_actions:
                        self.children[action] = Node(parent=self, last_action=action)
                else:
                    Node.deep_nodes.append(self)

        if parent is None: # start connecting the deep nodes
            print("Connecting deepest nodes")
            for node in Node.deep_nodes:
                previous_actions = []
                previous_node = node
                for i in range(Node.MAX_DEPTH - 2): # minus 2 to match the players
                    previous_actions.append(previous_node.last_action)
                    previous_node = previous_node.parent
                # Now find the Node that is similar to node but one depth less.
                next_node = self
                for i in range(Node.MAX_DEPTH - 2):
                    next_node = next_node.children[previous_actions.pop()]
                node.children = next_node.children
    
    def propagate_probability(self, probability):
        # the probability is a matrix with rows possible private info of self.player and
        # columns possible private info of the opponent. Each entry corresponds to the
        # conditional probability of arriving at that node given the private information.
        softmaxed = F.softmax(self.logits, dim=-1)
        for i, child in enumerate(self.children.values()): # correct order verified.
            give_prob = softmaxed[:, i].unsqueeze(1)
            new_probability = (give_prob * probability).t()
            child.probability += new_probability
            if not child.is_leaf:
                child.propagate_probability(new_probability)
    
    def reset_probability(self):
        if self.parent is not None:
            self.probability = torch.zeros([len(Die.faces) ** n for n in N_DICE])
            if self.player:
                self.probability = self.probability.t()
        if not self.is_leaf:
            for child in self.children.values():
                if child.depth > self.depth:
                    child.reset_probability()

    def who_wins(self, my_roll, opponent_roll):
        assert self.is_leaf
        all_dice = my_roll + opponent_roll
        claim = self.parent.last_action # This is my claim
        response = self.last_action # call C or doubt D. My opponent said this.
        face = claim[0]
        quantity = len(claim)
        true_count = all_dice.count(face)
        if face != "L":
            true_count += all_dice.count("L")
        if response == "C":
            if quantity == true_count:
                return 1 - self.player
            return self.player
        if response == "D":
            if true_count >= quantity:
                return self.player
            return 1 - self.player

    def compute_winner_matrix(self):
        assert self.is_leaf
        self.winner = torch.zeros([len(Die.faces) ** n for n in N_DICE]) # zeros or ones does not matter
        if self.player == 1:
            self.winner = self.winner.t()
        shape = self.winner.shape
        for i in range(shape[0]): # my roll index
            for j in range(shape[1]): # opponent roll index
                my_roll = Node.index2roll(i, N_DICE[self.player])
                opponent_roll = Node.index2roll(j, N_DICE[1 - self.player])
                winner = self.who_wins(my_roll, opponent_roll)
                self.winner[i, j] = winner


starter = np.random.choice(2)
players = [Player(i, human=(i + starter)%2, physical_dice=USE_PHYSICAL_DICE) for i in range(len(N_DICE))]
file_name = "grandfather.pkl"
if os.path.exists(file_name):
    # Open the file in binary read mode
    with open(file_name, 'rb') as file:
        # Load the pickle object
        grandfather_node = pickle.load(file)
    print("Pickle object loaded successfully.")
else:
    print("Could not find " + file_name)
    grandfather_node = Node(None)
Player.faceoff(players[0], players[1])


def get_prob_1_wins(leaf):
    prob_1_wins = leaf.probability * leaf.winner
    if leaf.player:
        prob_1_wins = prob_1_wins.t()
    return prob_1_wins

# def get_advantage(leaf, prob_1_wins=0.5):
#     score_1 = torch.sum(leaf.probability * leaf.winner)
#     score_0 = torch.sum(leaf.probability) - score_1
#     return(score_1 / prob_1_wins - score_0 / (1 - prob_1_wins))

def get_entropy(leaf):
    entropy = - leaf.probability * torch.log(leaf.probability) / np.log(2)
    if leaf.player:
        entropy = entropy.t()
    return entropy

def get_loss(player, balance_factor=0.01):
    stacked = torch.stack([get_prob_1_wins(leaf) for leaf in Node.leaves], dim=0)
    prob_1_wins = torch.sum(stacked) / (len(Die.faces) ** np.sum(N_DICE))
    loss = (-1) ** player * prob_1_wins

    # stacked = torch.Tensor([get_advantage(leaf, prob_1_wins) ** 2 for leaf in Node.leaves])
    stacked = torch.stack([- get_entropy(leaf) for leaf in Node.leaves], dim=0)
    loss += balance_factor * torch.sum(stacked)
    return loss

def calculate_equilibrium(lr = 0.005):
    opt_p0 = torch.optim.Adam(Node.player_logits[0], lr=lr)
    opt_p1 = torch.optim.Adam(Node.player_logits[1], lr=lr)

    for it in range(1500, 2000):
        balance_factor = 1 / (it + 1)

        opt_p0.zero_grad()
        grandfather_node.reset_probability()
        grandfather_node.propagate_probability(grandfather_node.probability)
        loss = get_loss(0, balance_factor)
        loss.backward()
        opt_p0.step()
        
        if it % 10 == 0:
            print(f"iter {it:4d}   loss_0 ≈ {loss.item(): .4f}")

        opt_p1.zero_grad()
        grandfather_node.reset_probability()
        grandfather_node.propagate_probability(grandfather_node.probability)
        loss = get_loss(1, balance_factor)
        loss.backward()
        opt_p1.step()

        if it % 10 == 0:
            print(f"iter {it:4d}   loss_1 ≈ {loss.item(): .4f}")

    # Save the object deeply
    with open('grandfather.pkl', 'wb') as f:
        pickle.dump(grandfather_node, f, protocol=pickle.HIGHEST_PROTOCOL)