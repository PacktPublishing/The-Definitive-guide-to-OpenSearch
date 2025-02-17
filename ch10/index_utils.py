from copy import deepcopy
import logging


BASE_SETTINGS = {
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
        "genres": {"type": "text",
                   "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                   }},
        "plot": {"type": "text"},
        "rating": {"type": "float"},
        "vote": {"type": "integer"},
        "revenue": {"type": "float"},
        "thumbnail": {"type": "keyword"},
        "directors": {"type": "text",
                      "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256}
                   }},
        "actors": {"type": "text",
                   "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                   }},
    }}}


def delete_then_create_index(os_client, index_name, pipeline_name, additional_fields):
  if os_client.indices.exists(index_name):
    logging.info(f'Deleting existing index {index_name}')
    os_client.indices.delete(index_name)

  settings = deepcopy(BASE_SETTINGS)
  settings['settings']['default_pipeline'] = pipeline_name
  settings['mappings']['properties'].update(additional_fields)
  os_client.indices.create(index_name, body=settings)
  logging.info(f'Created index {index_name}')