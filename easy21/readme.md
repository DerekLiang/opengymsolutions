# The problem
It is my solution to David Silver's Reinforcement Learning Assignment [Easy21](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf).

# Algorithm explanation
The action count data was stored in a 5-dimension matrices with mapping to dealer, player, action, new-dealer, new-player. It works in game with relative small total states. It won't work in OpenAI Gym CartPole problem, since the possible states are too big.


# Results
Here is my result in 3D visualization via Plotly.

<img src="./training-result.gif">