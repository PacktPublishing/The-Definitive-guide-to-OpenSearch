'''
Provides settings and a method to delete then create an index. The method takes
additional field definitions, allowing the scripts that use it to specify the
embedding field. Index create is also where the index is joined to the pipeline
with the default_pipeline index setting.

CAUTION! If the index exists, delete_then_create_index will delete it!
'''


from copy import deepcopy
import logging
import time


# The base mapping doesn't contain a knn field, or an embedding source field.
# The individual usage scripts (exact.py, e.g.) add the right kNN field, derived
# from the model.
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
  # Delete the existing index
  if os_client.indices.exists(index_name):
    logging.info(f'{int(time.time())}: Deleting existing index {index_name}')
    os_client.indices.delete(index_name)

  # Construct settings
  settings = deepcopy(BASE_SETTINGS)
  settings['settings']['default_pipeline'] = pipeline_name
  settings['mappings']['properties'].update(additional_fields)

  # Create the new index
  logging.info(f'{int(time.time())}: Creating index {index_name}')
  os_client.indices.create(index_name, body=settings)
