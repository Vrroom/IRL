import torch
import random
import PolicyGradient

class Human () :

    def __init__ (self, env) : 
        self.action = 1
        env.render()

        keyPress = lambda key, mod : self.keyPress(key, mod)
        keyRelease = lambda key, mod : self.keyRelease(key, mod)

        env.unwrapped.viewer.window.on_key_press = keyPress
        env.unwrapped.viewer.window.on_key_release = keyRelease

    def keyPress(self, key, mod) : 
        if key == 65363 : 
            self.action = 0
        elif key == 65361 : 
            self.action = 2

    def keyRelease(self, key, mod) : 
        self.action = 1

    def __call__ (self, *args, **kwargs):
        return self.action

class REINFORCE () : 

    def __init__ (self, path) :
        self.model = torch.load(path)

    def __call__ (self, observation) : 
        return PolicyGradient.getBestAction(self.model, observation)

