import json
from collections import OrderedDict


with open("movies_100k_LLM_generated.json", "r") as f:
  words = OrderedDict()
  for line in f:
    data = json.loads(line)
    title = data['title']
    if len(title.split(' ')) < 3:
      continue
    for word in title.split(' '):
      if word in words.keys():
        words[word] += 1
      else:
        words[word] = 1

  for item in (sorted(words.items(), key=(lambda x: x[1]))):
    print(f'{item[0]}: {item[1]}')
