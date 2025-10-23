class DataStorage:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._json_data = {}
        self._total_length = 0
        self._total_tokens = 0
    
    def add(self, id, json_item, length):
        self._json_data[id] = json_item
        self._total_length += length
        self._total_tokens += 1
    def get_data(self):
        return self._json_data.copy(),  self._total_length, self._total_tokens

storage = DataStorage()