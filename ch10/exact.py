"""
Exact k-nearest neighbor search implementation for movie recommendations.

This module provides functionality to:
1. Set up and run exact k-nearest neighbor (kNN) search on a movie dataset
2. Create and manage an OpenSearch index with vector embeddings
3. Run similarity searches using script scoring and optional filters

Usage:
    python exact.py [--skip-indexing] [--filtered]

Arguments:
    --skip-indexing: Skip index creation if it already exists
    --filtered: Run query with genre and rating filters

The script requires an OpenSearch cluster with the neural-search plugin installed.
For Amazon OpenSearch Service deployments, set port to 443.
"""


import argparse
from auto_incrementing_counter import AutoIncrementingCounter
from copy import deepcopy
import jsonpath_ng.ext
import index_utils
import logging
import model_utils
import movie_source
from os_client_factory import OSClientFactory
import opensearchpy.helpers


# NOTE: Much of the code is duplicated across the various examples. Better
# coding practice is to build modules/classes to encapsulate the duplicated code.
# We've constructed the examples this way to facilitate expositon in the book
# and for the examples to be self-contained


# Defines the index and pipelines created by the script.
INDEX_NAME = 'exact_movies'
PIPELINE_NAME = 'exact_pipeline'

# Set the bulk size. If your indexing requests are timing out, make this
# smaller.
BULK_SIZE = 1000
NUMBER_OF_MOVIES = movie_source.TOTAL_MOVIES
TOTAL_NUMBER_OF_BULKS = NUMBER_OF_MOVIES // BULK_SIZE

# You can try out other models to see how they behave for the movies data set.
# This script doesn't use remote models, but see model_utils.py for a list of
# models you can try.
MODEL_SHORT_NAME = "all-MiniLM-L12-v2"
MODEL_REGISTER_BODY = {
  "name": model_utils.DENSE_MODELS_HF[MODEL_SHORT_NAME]['name'],
  "model_format": "TORCH_SCRIPT",
  "version": model_utils.DENSE_MODELS_HF[MODEL_SHORT_NAME]['version']
}


# Defines the source text for the embedding field. You can modify
# movie_source.py to change how the data is treated. 
EMBEDDING_SOURCE_FIELD_NAME = 'embedding_source'
EMBEDDING_FIELD_NAME = 'embedding'
KNN_FIELDS = {
  "embedding": {
    "type": "knn_vector",
    "dimension": model_utils.DENSE_MODELS_HF[MODEL_SHORT_NAME]['dimensions']
  }
}


# Definition for the ingest pipeline. Maps the EMBEDDING_SOURCE_FIELD to the
# EMBEDDING_FIELD. OpenSearch neural plugin uses these fields for creating the
# embedding as you ingest data.
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


# An exact kNN query. Uses a match_all query along with a Painless script to
# compute the score.
script_query = {
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
}}}}}


# An exact kNN query with a bool filter for SciFi movies. First filters the
# movies for SciFi and then computes a score based on vector distance.
filtered_script_query = {
  "sort": [{"_score": "asc"}],
  "query": {
    "script_score": {
      "query": {
        "bool": {
          "filter": [
            { "term": {
                "genres.keyword": "Sci-Fi"
            }},
            { "range": {"rating": {"gte": 6.0}}}]
        }
      },
      "script": {
        "source": "knn_score",
        "lang": "knn",
        "params": {
          "field": EMBEDDING_FIELD_NAME,
          "query_value": [],
          "space_type": "l2"
}}}}}


# Main function. Finds or loads the embedding model, creates the index (unless
# --skip-indexing is a command-line paramater), creates an embedding for the
# query "A sweeping space opera about good and evil centered around a powerful
# family set in the future" and then runs the exact query and prints the search
# response.
def main(skip_indexing=False, filtered=False, user_query=None):
  logging.info(f"Query: {user_query}")

  # See os_client_factory.py for details on the set up for the opensearch-py
  # client.
  os_client = OSClientFactory().client()

  # Find an existing model, or register the model. Deploy the model to prepare
  # it for use. You need to _deploy the model whenever the cluster shuts down.
  # This can take some time. The code busy waits for the model to be deployed.
  logging.info(f"Finding or deploying model {MODEL_SHORT_NAME}")
  model_id = model_utils.find_or_deploy_model(
    os_client=os_client,
    model_name=model_utils.DENSE_MODELS_HF[MODEL_SHORT_NAME]['name'],
    body=MODEL_REGISTER_BODY
  )

  # If you did not disable indexing, this will create a new index, set up an
  # ingest pipeline for automatically generating vector embeddings on ingest,
  # read the movies data (movie_source.py) and send it to the index.
  if not skip_indexing:

    # Create an ingest pipeline
    pipeline_definition = deepcopy(ingest_pipeline_definition)
    pipeline_definition['processors'][0]['text_embedding']['model_id'] = model_id
    os_client.ingest.put_pipeline(id=PIPELINE_NAME, body=pipeline_definition)

    # Create an index with the pipeline
    logging.info(f"Creating index {INDEX_NAME}")
    index_utils.delete_then_create_index(
      os_client=os_client,
      index_name=INDEX_NAME,
      ingest_pipeline_name=PIPELINE_NAME,
      additional_fields=KNN_FIELDS
    )

    # Read and add documents to the index with the opensearch-py bulk helper.
    logging.info(f"Indexing documents")
    counter = AutoIncrementingCounter()
    for bulk in movie_source.bulks(BULK_SIZE, INDEX_NAME):
      logging.info(f"Indexing bulk {str(counter)} / {TOTAL_NUMBER_OF_BULKS}")
      opensearchpy.helpers.bulk(os_client, bulk, timeout=600, max_retries=10)
  else:
    logging.info(f"Skipping indexing")

  # Run a query. Calls the LLM to generate a vector embedding for the user query
  # (see model_utils.py) and then adds that embedding to the OpenSearch query.
  logging.info(f"Running query")
  if filtered:
    query = deepcopy(filtered_script_query)
  else:
    query = deepcopy(script_query)
  question = user_query if user_query else "Sci-fi about the force and jedis"
  query_embedding = model_utils.create_embedding(os_client, model_id, question)

  expr = jsonpath_ng.ext.parser.parse('query.script_score.script.params.query_value')
  query = expr.update(query, query_embedding)
  response = os_client.search(index=INDEX_NAME, body=query)

  # Print the search response.
  logging.info(f"Query response")
  for hit in response['hits']['hits']:
    logging.info(f"score: {hit['_score']}")
    logging.info(f"title: {hit['_source']['title']}")
    logging.info(f"plot: {hit['_source']['plot']}")
    logging.info(f"embedding source: {hit['_source'][EMBEDDING_SOURCE_FIELD_NAME]}\n")


if __name__ == "__main__":
  # Info level logging.
  logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

  parser = argparse.ArgumentParser(
      prog="main",
      description="Loads movie data, and runs exact kNN queries. Use --skip-indexing"
      " to skip the from-scratch creation of the index.",
  )
  parser.add_argument("--skip-indexing", default=False, action="store_true")
  parser.add_argument("--filtered", default=False, action="store_true")
  parser.add_argument("--query", default="Sci-fi about the force and jedis",
                      action="store")
  args = parser.parse_args()
  main(skip_indexing=args.skip_indexing,
       filtered=args.filtered,
       user_query=args.query)
