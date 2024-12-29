import os
import json


from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk


OPENSEARCH_HOST = os.environ.get('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = os.environ.get('OPENSEARCH_PORT', 9200)
OPENSEARCH_AUTH = (os.environ.get('OPENSEARCH_ADMIN_USER', 'admin'),
                   os.environ.get('OPENSEARCH_ADMIN_PASSWORD', ''))


os_client = OpenSearch(
  hosts = [{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
  http_auth = OPENSEARCH_AUTH,
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
  os_client.indices.create(index='movies', body=
  {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 1,
      "max_ngram_diff": 7,
      "analysis": {
        "filter": {
          "reverse_filter": {
            "type": "reverse"
          },
          "shingle_filter": {
            "type": "shingle",
            "min_shingle_size": 2,
            "max_shingle_size": 3
          }
        },
        "tokenizer": {
          "ngram_tokenizer": {
            "type": "ngram",
            "min_gram": 3,
            "max_gram": 10,
            "token_chars": ["letter", "digit" ]
          },
          "edge_ngram_tokenizer": {
            "type": "edge_ngram",
            "min_gram": 3,
            "max_gram": 10,
            "token_chars": ["letter", "digit" ]
          }
        },
        "analyzer": {
          "my_reverse_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": [ "lowercase", "reverse_filter" ]
          },
          "edge_ngram_analyzer": {
            "tokenizer": "edge_ngram_tokenizer",
            "filter": [ "lowercase" ]
          },
          "trigram_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": [
              "lowercase",
              "shingle"
            ]
          }
        }
      }
    },
    "mappings": {
      "properties": {
        "id": {"type": "integer"},
        "title": {"type": "text",
                  "copy_to": ["reverse_title", "completions_title", "sayt_title"],
                   "fields": {
                      "keyword": {"type": "keyword", 
                                  "ignore_above": 256},
                      "trigram": {
                        "type": "text",
                        "analyzer": "trigram_analyzer"
                      }
                   }},
        "reverse_title": {"type": "text",
                          "analyzer": "my_reverse_analyzer"},
        "sayt_title": {"type": "search_as_you_type"},
        "completions_title": {"type": "completion"},
        "year": {"type": "integer"},
        "duration": {"type": "integer"},
        "genres": {"type": "text",
                   "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                   }},
        "plot": {"type": "text"},
        "rating": {"type": "float"},
        "vote": {"type": "integer"},
        "revenue": {"type": "float"},
        "thumbnail": {"type": "keyword"},
        "directors": {"type": "text"},
        "actors": {"type": "text", "copy_to": ["completions_actors", "edge_ngram_actors"]},
        "completions_actors": {"type": "completion"},
        "edge_ngram_actors": {"type": "text", "analyzer": "edge_ngram_analyzer"}
    }
  }})

  with open('movies_100k_LLM_generated.json', 'r') as f:
    nline = 0
    buffer = []
    for line in f:
      if not line:
        break
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
