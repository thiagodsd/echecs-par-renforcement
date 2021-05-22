import math, random, util
from collections import defaultdict

class MDP:
    """ 
    An abstract class representing a Markov Decision Process (MDP)
    """
    
    def startState(self):
	    """ 
	    Return the start state.
	    """ 
    	raise NotImplementedError("Override me")

    
    def actions(self, state):
	    """ 
	    Return set of actions possible from |state|
	    """
	    raise NotImplementedError("Override me")

    
    def succAndProbReward(self, state, action): 
	    """
        Return a list of (newState, prob, reward) tuples corresponding
        to edges coming out of |state|. Mapping to notation from class:
          state    = s, 
          action   = a, 
          newState = s', 
          prob     = T(s, a, s'), 
          reward = Reward(s, a, s')
        If IsEnd(state), return the empty list.
	    """
    	raise NotImplementedError("Override me")

    def discount(self): 
		"""
		Return the discount  
		"""
    	raise NotImplementedError("Override me")

    
    def computeStates(self):
        """
        Compute set of states reachable from startState.  Helper function for
        MDPAlgorithms to know which states to compute values and policies for.
        This function sets |self.states| to be the set of all states.
        """
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print(self.states)


class RLAlgorithm:
    """
     Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
     to know is the set of available actions to take.  The simulator (see
     simulate()) will call getAction() to get an action, perform the action, and
     then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
     its parameters.
    """
    
    def getAction(self, state): raise NotImplementedError("Override me")
    """ Your algorithm will be asked to produce an action given a state."""
    
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")
    """ 
         We will call this function when simulating an MDP, and you should update parameters.
         If |state| is a terminal state, this function will be called with (s, a, 0, None). 
         When this function is called, it indicates that taking action |action| in state |state| 
         resulted in reward |reward| and a transition to state |newState|.
    """


class BlackjackMDP(MDP):
    """
    The BlackjackMDP class is a subclass of MDP that models the BlackJack game as a MDP
    """
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    def startState(self):
        """
         Return the start state.
         Each state is a tuple with 3 elements:
           -- The first element of the tuple is the sum of the cards in the player's hand.
           -- If the player's last action was to peek, the second element is the index
              (not the face value) of the next card that will be drawn; otherwise, the
              second element is None.
           -- The third element is a tuple giving counts for each of the cards remaining
              in the deck, or None if the deck is empty or the game is over (e.g. when
              the user quits or goes bust).
        """
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    def actions(self, state):
        """
        Return set of actions possible from |state|.
        You do not must to modify this function.
        """
        return ['Take', 'Peek', 'Quit']

    def succAndProbReward(self, state, action):
        """
        Given a |state| and |action|, return a list of (newState, prob, reward) tuples
        corresponding to the states reachable from |state| when taking |action|.
        A few reminders:
         * Indicate a terminal state (after quitting, busting, or running out of cards)
           by setting the deck to None.
         * If |state| is an end state, you should return an empty list [].
         * When the probability is 0 for a transition to a particular new state,
           don't include that state in the list returned by succAndProbReward.
        """
        total, index, deck = state

        if deck is not None:
            deck_index = [i for i, v in enumerate(list(deck)) if v > 0]
            prob = {i : deck[i]/sum(deck) if deck[i] != sum(deck) else 1 for i in deck_index}

            if action == 'Take':
                if index is None:
                    new_state = {}
                    for i in deck_index:
                        new_total = total + self.cardValues[i]
                        reward = total
                        if new_total > self.threshold:
                            new_deck = None
                            reward = 0
                        else:
                            new_deck = tuple([card-1 if pos == i else card \
                                              for pos, card in enumerate(deck)])
                            if not [i for i, v in enumerate(list(new_deck)) if v > 0]:
                                new_deck = None
                                reward = new_total
                        new_state[(new_total, None, new_deck)] = prob[i]
                else:
                    prob = 1
                    new_total = total + self.cardValues[index]
                    reward = total
                    if new_total > self.threshold:
                        new_deck = None
                        reward = 0
                    else:
                        new_deck = tuple([card-1 if pos == index else card \
                                          for pos, card in enumerate(deck)])
                        if not [i for i, v in enumerate(list(new_deck)) if v > 0]:
                            new_deck = None
                            reward = new_total
                    new_state = {(new_total, None, new_deck): prob}
                return [(ns, new_state[ns], reward) for ns in new_state]

            if action == 'Peek':
                if index is None:
                    return [((total, i, deck), prob[i], -self.peekCost) for i in deck_index]
                return []
            return [((total, None, None), 1, total)]
        return []

    def discount(self):
        """
        Return the descount  that is 1
        """
        return 1

class QLearningAlgorithm(RLAlgorithm):
    """
    Performs Q-learning.  Read util.RLAlgorithm for more information.
    actions: a function that takes a state and returns a list of actions.
    discount: a number between 0 and 1, which determines the discount factor
    featureExtractor: a function that takes a state and action and returns a
    list of (feature name, feature value) pairs.
    explorationProb: the epsilon value indicating how frequently the policy
    returns a random action
    """
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    def getQ(self, state, action):
        """
         Return the Q function associated with the weights and features
        """
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        Produce an action given a state, using the epsilon-greedy algorithm: with probability
        |explorationProb|, take a random action.
        """
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    def getStepSize(self):
        """
        Return the step size to update the weights.
        """
        return 1.0 / math.sqrt(self.numIters)

    def incorporateFeedback(self, state, action, reward, newState):
        """
         We will call this function with (s, a, r, s'), which you should use to update |weights|.
         You should update the weights using self.getStepSize(); use
         self.getQ() to compute the current estimate of the parameters.

         HINT: Remember to check if s is a terminal state and s' None.
        """
        # BEGIN_YOUR_CODE
        if newState is not None:
            alpha = 50/(50 + self.getStepSize())
            for i, f in self.featureExtractor(state, action):
                self.weights[i] = self.weights[i] + alpha * (reward + \
                                  self.discount*self.getQ(newState, action) \
                                             - self.getQ(state, action))*f
        # END_YOUR_CODE

def identityFeatureExtractor(state, action):
    """
    Return a single-element list containing a binary (indicator) feature
    for the existence of the (state, action) pair.  Provides no generalization.
    """
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)