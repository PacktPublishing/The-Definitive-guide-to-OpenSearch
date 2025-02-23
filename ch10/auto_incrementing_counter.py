'''
A counter class that automatically increments when converted to string.

This class provides a counter that increments automatically when str() or repr() 
is called on it. This makes it convenient for use in logging statements where you 
want to track the count of operations without explicitly incrementing a counter.

Attributes:
    _count (int): The current count value
    _increment (int): The amount to increment by each time

Methods:
    increment(amount=None): Increments counter by specified amount or default increment
    reset(): Resets counter back to 0
    count: Property that returns current count value
    
Example:
    counter = AutoIncrementingCounter()
    logging.info(f"Processing bulk {counter}")  # Prints 1
    logging.info(f"Processing bulk {counter}")  # Prints 2

Note, this makes __str__ and __repr__ mutating, which is not Pythonic, but is
convenient to just stick it in a logging.info call and not have to deal with
incrementing some counter in the loop
'''


class AutoIncrementingCounter:

  def __init__(self, initial_value=0, initial_increment=1):
    self._count = initial_value
    self._increment = initial_increment

  def __str__(self):
    self.increment()
    return str(self._count)

  def __repr__(self):
    return str(self)

  @property
  def count(self):
    return self._count

  def increment(self, amount=None):
    if amount is None:
      amount = self._increment
    self._count += amount
    return self._count

  def reset(self):
    self._count = 0
    return self._count
