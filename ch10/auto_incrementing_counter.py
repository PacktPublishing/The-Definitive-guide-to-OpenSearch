'''
Utility class to take care of some of the mess around logging with a count of
bulks sent. Starts at 0. Increments automatically on __str__ or __repr__
reference

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
