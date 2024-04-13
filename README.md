# completesnakegameai

Dependencies:

```
sudo dnf install python3-pip python3-numpy python3-pygame python3-ipython
pip3 install torch
```

Structure of repo:

```
----- game_human.py: vanilla game
----- game_ai.py: AI version of the game, with bindings for AI inputs and SPEED var changed.
----- model.py: Main script for reinforcement learning QNet code.
----- train.py: calls model.py, trains model with checkpoints saved to disk and multiple threads
----- inference.py: script to load last checkpoint and run the game from that without any learning
```
