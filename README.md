# Text Generator
Creates text based on Shakespeare or any other text you want to.

## How to use it

if this is the first time you run it, verify that lines 40-45 are commented.

- execute ```python main.py```, it will ask you to leave the input empy if you want to use the dafault training dataset (Shakespeare's Romeo and Juliet) 
or choose your own file (spported formats txt and rtf).

- let the model runt it's training, if it's taking too long you can change the number of epochs on the `constants.py` file

- After the training is done, run again ```python main.py``` but this time comment line 38 and uncomment lines 40-45.

- the prompt will ask you to enter a string and will create a new output based on that, in the style of the text you povided as training.

That's it! 🎉
