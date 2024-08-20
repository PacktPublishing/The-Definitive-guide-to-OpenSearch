from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
import json


os_client = OpenSearch(
  hosts = [{'host': 'search-prashagr-def-guide-lslkr2b22fkoztdtlenuu2hro4.us-east-1.es.amazonaws.com', 'port': 443}],
  http_auth = ('admin', 'DefGuide123!'),
  use_ssl = True,
  verify_certs = False,
  ssl_assert_hostname = False,
  ssl_show_warn = False,
)


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
  return data


if __name__=='__main__':
  os_client.indices.delete(index='movies', ignore=[400, 404])
  os_client.indices.create(index='movies', body={
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 1
    },
    "mappings": {
      "properties": {
        "id": {"type": "integer"},
        "title": {"type": "text"},
        "year": {"type": "integer"},
        "duration": {"type": "integer"},
        "genres": {"type": "text"},
        "plot": {"type": "text"},
        "rating": {"type": "float"},
        "vote": {"type": "integer"},
        "revenue": {"type": "float"},
        "thumbnail": {"type": "keyword"},
        "directors": {"type": "text"},
        "actors": {"type": "text"}
      }
    }
  })

  with open('movies_100k_LLM_generated.json', 'r') as f:
    nline = 0
    buffer = []
    for line in f:
      data = json.loads(line)
      buffer.append(
          { 
            "_op_type": "create",
            "_index": 'movies',
            "_source": clean_data(data)
          }
        )
      nline += 1
      if nline % 5000 == 0:
        print(nline, ' lines processed')
        bulk(os_client, buffer)
        buffer = []
