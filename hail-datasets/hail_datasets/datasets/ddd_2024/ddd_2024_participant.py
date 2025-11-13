import os
import numpy as np


class Participant():
    def __init__(self, name, base_index_dir):
        self.name = name
        self.index_path = os.path.join(base_index_dir, name)
        # TODO: Add metadata
    
    def __str__(self):
        return f"{self.name}"
    

class CollatedParticipant():
    def __init__(self, list_of_participants, config):
        self.name = np.array([x.name for x in list_of_participants])
        self.index_path = np.array([x.index_path for x in list_of_participants])

    def __str__(self):
        return f"(CollatedParticipant Batch Size: {self.name.shape[0]})"
