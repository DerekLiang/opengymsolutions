import numpy as np

#assignment description at http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf

# for player 0 represents terminal state, 1-21 represents the range of player card's sum
# for dealer 0 represents terminal state, 1-21 represents the range of dealers card's sum

def __drawABlackCard():
    '''black card are 11-20'''
    return np.random.randint(11,21)

def __drawARedCard():
    '''black card are 11-20'''
    return np.random.randint(1,11)

def __drawACard():
    '''red card are 1-10 black card are 11-20'''
    if np.random.randint(0,3)==2:
        return __drawARedCard()
    return __drawABlackCard()

def __toValue(card):
    return (card-10) if card > 10 else (-(card-10))

def __drawABlackCardByValue():
    card = __drawABlackCard()
    return __toValue(card)

def __drawACardByValue():
    card = __drawACard()
    return __toValue(card)

def __print(array):
    rx, ry = array.shape
    s = ''
    for x in range(ry):
        s += "{:8} ".format(x)
    for x in range(rx):
        s += '\n'
        for y in range(ry):
            if y == 0:
                s += "{:2} ".format(x)
            s += "{:8.3f} ".format(array[x,y])

    print(s)

def step(state=(None,None), action=None):
    ''' state is (dealer card, player sum)
        action is string of value: 0 or 1
        return a tuple (dealer sum, player sum, isTerminated, reward)
    '''
    dealer, player = state

    if dealer is None and player is None:
        return (__drawABlackCardByValue(), __drawABlackCardByValue())

    if action == 0: # 'hit'
        player += __drawACardByValue()
        if player>21 or player<1:
            return (dealer, 0, True, -1)
        else:
            return (dealer, player, False, 0)
    elif action == 1: #'stick'
        while True:
            dealer += __drawACardByValue()
            if dealer>=17:
                if dealer>21:
                    dealer = 0
                    reward = 1
                elif dealer==player:
                    reward = 0
                elif dealer<player:
                    reward = 1
                else:
                    dealer = 0
                    reward = -1
                return (dealer, player, True, reward)
            elif dealer<1:
                return (0, player, True, -1)

    assert('unknow action')

values = np.zeros((22, 22))
visitedCount = np.zeros((22, 22))
actionCount = np.zeros((22, 22, 2, 22, 22)) # this is state0 -> action0 -> state1 counter mapper.

N0 = 100

for i in range(5000):
    dealer, player = step() # start new hand

    isTerminated = False

    while not isTerminated:
        e = 100/(100 + visitedCount[dealer, player])
        if np.random.rand()<=e:
            action = np.random.randint(2)
        else:
            action = np.argmax([ ((values*actionCount[dealer, player, x])/(actionCount[dealer, player, x].sum())).sum() for x in range(2) ])

        newDealer, newPlayer, isTerminated, reward =  step((dealer, player), action)

        visitedCount[dealer, player] += 1
        actionCount[dealer, player, action, newDealer, newPlayer] += 1
        values[dealer, player] += (reward - values[dealer, player])/visitedCount[dealer, player]

        dealer, player = newDealer, newPlayer

# __print(values[1:,1:][:10,:])
__print(values)

# for i in range(1):
#     for x in range(1,22):
#         for y in range(1, 11):

