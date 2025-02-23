class AutoIncrementingCounter:
    def __init__(self, initial_value=0):
        self._count = initial_value
    
    @property
    def count(self):
        return self._count
    
    def increment(self, amount=1):
        self._count += amount
        return self._count
    
    def reset(self):
        self._count = 0
        return self._count

    def __str__(self):
        self.increment
        return str(self._count)