"""
Movie Data Processing Module

This module provides functionality for processing and normalizing movie data
from a JSON file, preparing it for bulk indexing into OpenSearch.

Key Components: - Data cleaning and normalization for movie records - Safe type
conversion for numeric fields - String list processing for fields like genres,
actors, directors - Creation of an embedding source field from movie metadata -
Generator functions for streaming movie data processing

Generator Functions:
    movies(): Yields normalized movie records one at a time bulks(n_movies,
    index_name): Yields batches of n movies formatted for bulk indexing

Utility Functions:
    safe_int(val): Safely converts values to integers safe_float(val): Safely
    converts values to floats split_and_strip_whitespace(str): Processes
    comma-separated strings into clean lists clean_data(data): Normalizes and
    enriches raw movie data

The module expects input from a 'movies_100k_LLM_generated.json' file with one
JSON movie record per line.
"""


import json


MOVIES_FILE_PATH = 'movies_reduced.ndjson'
TOTAL_MOVIES = 10000


# Treatment of numbers. 
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


# For treatment of string lists, removes commas (via split()), leading and
# trailing whitespace from the tokens.
def split_and_strip_whitespace(str):
  lis = str.split(',')
  return [x.strip() for x in lis]


# Takes a raw JSON line from the input file, and processes numeric and array
# data to normalize. Creates an embedding_source field composed of title, plot,
# and genres
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
  embedding_source += f' movie genres: {' '.join(data["genres"])}'
  embedding_source += f' movie plot: {data["plot"]} '
  data['embedding_source'] = " ".join(embedding_source.split()[:500])
  return data


# Generator the produces one normalized movie as a json dict at a time
def movies():
  with open(MOVIES_FILE_PATH, 'r') as f:
    for line in f:
      if not line:
        break
      data = clean_data(json.loads(line))
      yield data


# Generator that produces one bulk body. Use n-movies to tune for the bulk
# timeout
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