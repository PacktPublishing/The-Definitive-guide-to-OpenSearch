import json
import logging


def safe_int(val):
  if not val:
    return 0
  try:
    return int(val)
  except TypeError:
    return 0
  except ValueError:
    return 0


def safe_float(val):
  if not val:
    return 0.0
  try:
    return float(val)
  except TypeError:
    return 0.0
  except ValueError:
    return 0.0


def split_and_strip_whitespace(str):
  lis = str.split(',')
  return [x.strip() for x in lis]


def clean_data(data):
  data['id'] = safe_int(data['id'])
  data['year'] = safe_int(data['year'])
  data['duration'] = safe_int(data['duration'])
  data['like'] = safe_int(data['like'])
  data['rating'] = safe_float(data['rating'])
  data['genres'] = split_and_strip_whitespace(data['genres'])
  data['actors'] = split_and_strip_whitespace(data['actors'])
  data['directors'] = split_and_strip_whitespace(data['directors'])
  data['revenue'] = safe_float(data['revenue'])
  # Construct a source field for embeddings with information from the title, plot
  # and genres. Truncate at 500 tokens
  embedding_source = f'movie title: {data["title"]} '
  embedding_source += f'movie plot: {data["plot"]} '
  embedding_source += f'movie genres: {' '.join(data["genres"])}'
  data['embedding_source'] = " ".join(embedding_source.split()[:500])
  return data


def movies():
  with open('movies_100k_LLM_generated.json', 'r') as f:
    for line in f:
      if not line:
        break
      data = clean_data(json.loads(line))
      yield data


def bulks(n_movies, index_name):
  buffer = []
  for movie in movies():
    buffer.append(
        { 
          "_op_type": "create",
          "_index": index_name,
          "_source": movie
        }
      )
    if len(buffer) >= n_movies:
      yield buffer
      buffer = []
  if buffer:
    yield buffer