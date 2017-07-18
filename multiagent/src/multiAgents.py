# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # Ghosts
        uGhosts = 0
        minDistanceAway = 3   # decrease utility when ghosts are three steps or closer
        for ghostState in newGhostStates:
            uGhost = util.manhattanDistance(newPos, ghostState.getPosition())
            uGhost = (min(minDistanceAway,uGhost) - minDistanceAway) * 999
            uGhosts += uGhost
        
        # Food
        uFood = 0
        foodList = newFood.asList()
        for food in foodList:
            uFood += 1/float(util.manhattanDistance(newPos,food))   # discounted rewards for being further away
        
        score = successorGameState.getScore()
        utility = score + uFood + uGhosts
        return utility
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        
    def isTerminal(self, depth, gameState):
        return depth == self.depth \
            or gameState.isWin() \
            or gameState.isLose()
            
    def value(self, gameState, \
              searchType, searchState, \
              alphaBeta = (float('-inf'), float('inf'))):
        depth, agentIndex = searchState
        numAgents = gameState.getNumAgents()
        if agentIndex == numAgents:
            agentIndex %= numAgents
            depth += 1
        if self.isTerminal(depth, gameState):
            return self.evaluationFunction(gameState)
        if agentIndex == self.index:
            return self.maxValue(gameState, searchType, (depth, agentIndex), alphaBeta)
        else:
            if searchType == 'expectimax':
                return self.expValue(gameState, searchType, (depth, agentIndex))
            else:
                return self.minValue(gameState, searchType, (depth, agentIndex), alphaBeta)

    def maxValue(self, gameState, \
                 searchType, searchState, \
                 alphaBeta = (float('-inf'), float('inf'))):
        # Note: returns action instead of max value if root node (start state)
        depth, agentIndex = searchState
        alpha, beta = alphaBeta
        max_v = float('-inf')
        bestAction = None
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.value(successor, searchType, (depth, agentIndex + 1), (alpha, beta))
            if v > max_v:
                max_v = v
                bestAction = action
            if searchType == 'alphabeta':
                if max_v > beta: break
                alpha = max(alpha, v)
        if (depth, agentIndex) == (0,0):    # root
            return bestAction
        return max_v

    def minValue(self, gameState, \
                 searchType, searchState, \
                 alphaBeta):
        depth, agentIndex = searchState
        alpha, beta = alphaBeta
        min_v = float('inf')
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            min_v = min(min_v, self.value(successor, searchType, (depth, agentIndex + 1), (alpha, beta)))
            if searchType == 'alphabeta':
                if min_v < alpha: break
                beta = min(beta, min_v)
        return min_v

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        # initialize search tree and get best action
        searchType = 'minimax'
        searchState = (0, self.index)
        return self.maxValue(gameState, searchType, searchState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        # initialize search tree and get best action
        searchType = 'alphabeta'
        searchState = (0, self.index)
        return self.maxValue(gameState, searchType, searchState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # initialize search tree and get best action
        searchType = 'expectimax'
        searchState = (0, self.index)
        return self.maxValue(gameState, searchType, searchState)
        
    def expValue(self, gameState, \
        searchType, searchState):
        depth, agentIndex = searchState
        v = 0
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v += self.value(successor, searchType, (depth, agentIndex + 1))
        v /= float(len(actions))
        return v

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    # evaluation of a game state will be based on various utilities
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    
    # constants to be used in balancing utilities -- a lot of guesswork and magic numbers
    foodConstant = 1
    capsuleConstant = 20
    ghostConstant = 100
    
    # Food
    # the utility of food will be reduced the further away it is from pacman
    uFood = 0
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    for food in foodList:
        uFood += foodConstant/float(util.manhattanDistance(position,food))
    
    # Capsules
    # the utility of capsules will be reduced the further away it is from pacman
    uCapsules = 0
    capsules = currentGameState.getCapsules()
    for capsule in capsules:
        uCapsules += capsuleConstant/float(util.manhattanDistance(position,capsule))
    
    # Scared ghosts
    # when ghosts are scared (edible), the utility of ghosts will be reduced the further away it is from pacman
    uGhosts = 0
    ghostStates = currentGameState.getGhostStates()
    for ghostIndex, ghostState in enumerate(ghostStates):
        if (ghostState.scaredTimer):
            ghostPosition = ghostStates[ghostIndex].getPosition()
            uGhosts += ghostConstant/float(util.manhattanDistance(position,ghostPosition))
    
    # sum estimated utilities and return
    utility = score + uFood + uCapsules + uGhosts
    return utility

# Abbreviation
better = betterEvaluationFunction

