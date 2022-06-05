import json
from .State import State

class Abstraction:
    def __init__(self, json):
        if json["name"] == "hashtag":
            self._name = "#"
        else:
            self._name = json["name"]
        self._json          = json
        self._generated_by  = None
        self._states        = {}
        self.create_states()
    
    def create_states(self):
        
        for name in self._json["states"]:
            json               = self._json["states"][name]
            state              = State(name, json)
            self._states[name] = state
            if "q0" in name:
                self._current_state = state
    
    def name(self):
        return self._name

    def set_current_state(self, state):
        self._current_state = state

    def get_current_state(self):
        return self._current_state

    def states(self):
        return self._states
    
    def get_state(self, state):
        name = self.format_state_name(state)
        return self._states[name]
    
    def initial_state(self):
        try:
            name = self.format_state_name("q0")
            return self._states[name]
        except KeyError:
            name = self.format_state_name("q0")

    def format_state_name(self, state_name):
        return "<"+self._name+">-"+state_name

    def toJSON(self):
        return json.dumps(self)
    
    