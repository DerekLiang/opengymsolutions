import numpy as np

#assignment description at http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
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

def __drawACardByValue():
    card = __drawACard()
    return (card-10) if card > 10 else (-(card-10))

def step(state=(None,None), action=None):
    ''' state is (dealer card, player sum)
        action is string of value: 0 or 1
        return a tuple (dealer card, player sum, isTerminated, reward)
    '''
    dealer, player = state

    if dealer is None and player is None:
        return (__drawABlackCard(), __drawABlackCard())

    if action == 0: # 'hit'
        player += __drawACardByValue()
        if player>21 or player<1:
            return (dealer, player, True, -1)
        else:
            return (dealer, player, False, 0)
    elif action == 1: #'stick'
        while True:
            dealer += __drawACardByValue()
            if dealer>=17:
                if dealer>21:
                    reward = -1
                elif dealer==player:
                    reward = 0
                elif dealer<player:
                    reward = 1
                else:
                    reward = -1
                return (dealer, player, True, reward)
            elif dealer<1:
                return (dealer, player, True, -1)

    assert('unknow action')


values = np.zeros((21,21))
visitedCount = np.zeros((21,21))
actions = np.zeros((21,21,2,21,21)) # this is state0 -> action0 -> state1 counter mapper.

N0 = 100

for i in range(100):
    dealer, player = step() # start new hand

    isTerminated = False

    while not isTerminated:
        e = 100/(100 + visitedCount[dealer, player])
        if np.random.rand()<=e:
            action = np.random.randint(2)
        else:
            action = np.argmax([ ((values*actions[21,21,x])/actions[21,21,x].sum()).sum() for x in range(2) ])

        visitedCount[dealer, player] += 1

        newDealer, newPlayer, isTerminated, reward =  step((dealer, player), action)

        if not isTerminated:
            actions[dealer, player, action, newDealer, newPlayer] += 1

        values[dealer, player] += (reward - values[dealer, player])/visitedCount[dealer, player]

        dealer, player = newDealer, newPlayer

print(values)