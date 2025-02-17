import argparse
from copy import deepcopy
import jsonpath_ng.ext
import index_utils
import logging
import model_utils
import movie_source
import os
from os_client_factory import OSClientFactory
import opensearchpy.helpers
import time


OPENSEARCH_HOST = os.environ.get('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = os.environ.get('OPENSEARCH_PORT', 9200)
OPENSEARCH_AUTH = (os.environ.get('OPENSEARCH_ADMIN_USER', 'admin'),
                   os.environ.get('OPENSEARCH_ADMIN_PASSWORD', ''))
INDEX_NAME = 'exact_movies'
PIPELINE_NAME = 'exact_pipeline'


MODEL_SHORT_NAME = "msmarco-distilbert-base-tas-b"
MODEL_REGISTER_BODY = {
  "name": model_utils.HUGGING_FACE_MODELS[MODEL_SHORT_NAME]['name'],
  "model_format": "TORCH_SCRIPT",
  "version": model_utils.HUGGING_FACE_MODELS[MODEL_SHORT_NAME]['version']
}


EMBEDDING_SOURCE_FIELD_NAME = 'embedding_source'
EMBEDDING_FIELD_NAME = 'embedding'
KNN_FIELDS = {
  "embedding": {
    "type": "knn_vector",
    "dimension": model_utils.HUGGING_FACE_MODELS[MODEL_SHORT_NAME]['dimensions']
  }
}


ingest_pipeline_definition = {
  "description": "Embedding pipeline",
  "processors": [
    {
      "text_embedding": {
        "model_id": "",
        "field_map": {
          EMBEDDING_SOURCE_FIELD_NAME: EMBEDDING_FIELD_NAME
        }
      }
    }
  ]
}


script_query = {
 "size": 4,
 "query": {
   "script_score": {
     "query": {
       "match_all": {}
     },
     "script": {
       "source": "knn_score",
       "lang": "knn",
       "params": {
         "field": EMBEDDING_FIELD_NAME,
         "query_value": [],
         "space_type": "cosinesimil"
       }
     }
   }
 }
}

def main(skip_indexing=False):
  import json
  logging.basicConfig(level=logging.INFO)
  os_client = OSClientFactory().client()

  # Deploy the model
  logging.info(f"{int(time.time())}: Deploying {MODEL_SHORT_NAME}")
  model_id = model_utils.find_or_deploy_model(
    os_client=os_client,
    model_name=model_utils.HUGGING_FACE_MODELS[MODEL_SHORT_NAME]['name'],
    body=MODEL_REGISTER_BODY
  )
  logging.info(f"{int(time.time())}: Model {MODEL_SHORT_NAME} deployed with id {model_id}")

  if not skip_indexing:
    # Create a pipeline
    pipeline_definition = deepcopy(ingest_pipeline_definition)
    pipeline_definition['processors'][0]['text_embedding']['model_id'] = model_id
    os_client.ingest.put_pipeline(id=PIPELINE_NAME, body=pipeline_definition)

    # Create an index with the pipeline
    logging.info(f"{int(time.time())}: Creating index {INDEX_NAME}")
    index_utils.delete_then_create_index(
      os_client=os_client,
      index_name=INDEX_NAME,
      pipeline_name=PIPELINE_NAME,
      additional_fields=KNN_FIELDS
    )
    logging.info(f"{int(time.time())}: Indexing documents")
    for bulk in movie_source.bulks(1000, INDEX_NAME):
      opensearchpy.helpers.bulk(os_client, bulk, timeout=600, max_retries=10)
  else:
    logging.info(f"{int(time.time())}: Skipping indexing")

  # Run a query
  logging.info(f"{int(time.time())}: Running query")
  query = deepcopy(script_query)
  question = "A sweeping space opera about good and evil centered around a powerful family set in the future"
  query_embedding = model_utils.create_embedding(os_client, model_id, question)
  expr = jsonpath_ng.ext.parser.parse('query.script_score.script.params.query_value')
  query = expr.update(query, query_embedding)
  response = os_client.search(index=INDEX_NAME, body=query, size=10)

  for hit in response['hits']['hits']:
    logging.info(f"{int(time.time())}: title: {hit['_source']['title']}\nplot: {hit['_source']['plot']}\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      prog="main",
      description="Loads movie data, and runs exact kNN queries. Use --skip-indexing"
      " to skip the from-scratch creation of the index.",
  )
  parser.add_argument("-d", "--skip-indexing", default=False, action="store_true")
  main(skip_indexing=parser.parse_args().skip_indexing)
