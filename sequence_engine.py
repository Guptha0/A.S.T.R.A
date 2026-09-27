class ExperimentSequence:
    def __init__(self):
        """
        Initializes the state machine for the HAR protocol.
        Enforces a linear sequence of events.
        """
        self.states = ["IDLE", "BOTTLE_GRABBED", "CUP_GRABBED", "PROTOCOL_COMPLETE"]
        self.current_state_idx = 0
        
        # Define the exact action required to transition out of each state
        self.expected_actions = {
            "IDLE": "grabbed_bottle",
            "BOTTLE_GRABBED": "grabbed_cup"
        }
        
    def get_state(self):
        """Returns the current state string."""
        return self.states[self.current_state_idx]
        
    def update_state(self, action):
        """
        Receives an action string and checks it against the linear protocol.
        
        Returns:
            status (str): "success", "violation", "ignored", or "complete"
            message (str): Debugging / UI feedback string
        """
        current_state = self.get_state()
        
        if current_state == "PROTOCOL_COMPLETE":
            return "ignored", "Protocol already complete."
            
        expected_action = self.expected_actions.get(current_state)
        
        # Case 1: Correct action for the current state
        if action == expected_action:
            self.current_state_idx += 1
            new_state = self.get_state()
            if new_state == "PROTOCOL_COMPLETE":
                return "success", f"Protocol completed successfully!"
            return "success", f"Step verified. Advanced to {new_state}."
            
        # Case 2: Out of order action (a valid action, but at the wrong time)
        elif action in self.expected_actions.values():
            return "violation", f"Protocol violation. Expected '{expected_action}', got '{action}'."
            
        # Case 3: Random or unmonitored action (ignored)
        return "ignored", ""

if __name__ == "__main__":
    # Dummy tests
    engine = ExperimentSequence()
    print("Initial State:", engine.get_state())
    
    # Test violation
    status, msg = engine.update_state("grabbed_cup")
    print(f"Action 'grabbed_cup' -> Status: {status} | Msg: {msg}")
    
    # Test success 1
    status, msg = engine.update_state("grabbed_bottle")
    print(f"Action 'grabbed_bottle' -> Status: {status} | Msg: {msg}")
    
    # Test success 2
    status, msg = engine.update_state("grabbed_cup")
    print(f"Action 'grabbed_cup' -> Status: {status} | Msg: {msg}")
