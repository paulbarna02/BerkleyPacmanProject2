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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        score = 0

        for ghostPos in successorGameState.getGhostPositions():
            if manhattanDistance(newPos, ghostPos) < 2:
                score -= 1000

        if len(newFood.asList()) == 0:
            score += 1000
        else:
            closeFood = 1000
            for food in newFood.asList():
                closeFood = min(closeFood, manhattanDistance(newPos, food))
            score += 1.0 / closeFood

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def MinimaxRec(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), []
            bestAction = []
            bestScore = 0
            if agentIndex == 0:
                bestScore = -10000
                bestAction = []
                for x in gameState.getLegalActions(0):
                    score, action = MinimaxRec(gameState.generateSuccessor(0, x), 1, depth)
                    if score > bestScore:
                        bestScore = score
                        bestAction = x
            elif agentIndex < gameState.getNumAgents() - 1:
                bestScore = 10000
                bestAction = []
                for x in gameState.getLegalActions(agentIndex):
                    score, action = MinimaxRec(gameState.generateSuccessor(agentIndex, x), agentIndex + 1, depth)
                    if score < bestScore:
                        bestScore = score
                        bestAction = x
            elif depth != self.depth:
                bestScore = 10000
                bestAction = []
                for x in gameState.getLegalActions(agentIndex):
                    score, action = MinimaxRec(gameState.generateSuccessor(agentIndex, x), 0, depth + 1)
                    if score < bestScore:
                        bestScore = score
                        bestAction = x

            return bestScore, bestAction

        bestScore, bestMove = MinimaxRec(gameState,0,0)
        return bestMove



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(gameState, agentIndex, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), []
            bestAction = []
            bestScore = 0
            if agentIndex == 0:
                bestScore = -10000
                bestAction = []
                for x in gameState.getLegalActions(0):
                    score, action = alphaBeta(gameState.generateSuccessor(0, x), 1, alpha, beta, depth)
                    if score > bestScore:
                        bestScore = score
                        bestAction = x
                    if bestScore > beta:
                        break
                    alpha = max(alpha, bestScore)
            elif agentIndex < gameState.getNumAgents() - 1:
                bestScore = 10000
                bestAction = []
                for x in gameState.getLegalActions(agentIndex):
                    score, action = alphaBeta(gameState.generateSuccessor(agentIndex, x), agentIndex + 1, alpha, beta, depth)
                    if score < bestScore:
                        bestScore = score
                        bestAction = x
                    if bestScore < alpha:
                        break
                    beta = min(beta, bestScore)
            elif depth != self.depth:
                bestScore = 10000
                bestAction = []
                for x in gameState.getLegalActions(agentIndex):
                    score, action = alphaBeta(gameState.generateSuccessor(agentIndex, x), 0, alpha, beta, depth + 1)
                    if score < bestScore:
                        bestScore = score
                        bestAction = x
                    if bestScore < alpha:
                        break
                    beta = min(beta, bestScore)

            return bestScore, bestAction

        bestScore, bestMove = alphaBeta(gameState,0,-10000, 10000, 0)
        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(gameState, agentIndex, depth):
            nextIndex = 0

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), []
            elif agentIndex == gameState.getNumAgents() - 1:
                depth += 1
                nextIndex = 0
            else:
                nextIndex = agentIndex + 1


            bestAction = []
            value = 0

            if agentIndex == 0:
                value = -10000
            else:
                value = 0

            for x in gameState.getLegalActions(agentIndex):
                expectedMax = expectiMax(gameState.generateSuccessor(agentIndex, x), nextIndex, depth)[0]
                if agentIndex == 0:
                    if expectedMax > value:
                        value = expectedMax
                        bestAction = x
                else:
                    value = value + ((1.0 / len(gameState.getLegalActions(agentIndex))) * expectedMax)

            return value, bestAction

        bestScore, bestMove = expectiMax(gameState, 0, 0)
        return bestMove

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = 0
    if currentGameState.isLose():
        return -10000000000
    elif currentGameState.isWin():
        return 10000000000
    else:
        score += 1000 * currentGameState.getScore()
        for x in range(0, currentGameState.getNumFood()):
            score +=  1000.0 / manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getFood().asList()[x])
        ghostScore = 0
        goodGhost = []
        badGhost = []
        scaredTime = [ghost.scaredTimer for ghost in currentGameState.getGhostStates()]
        for x in range(len(scaredTime)):
            if scaredTime[x] > 0:
                ghostScore += 10 / manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(x + 1))
            else:
                ghostScore += manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(x + 1))
        score += ghostScore

        capsules = currentGameState.getCapsules()
        for x in capsules:
            score += 10 / manhattanDistance(x, currentGameState.getPacmanPosition())

        return score

# Abbreviation
better = betterEvaluationFunction
