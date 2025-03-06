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


# Defines the index and pipelines created by the script. If you change these
# here, you'll need to change the other example scripts to use the correct index
# and pipeline!
INDEX_NAME = 'approximate_movies_sq'
PIPELINE_NAME = 'approximate_sq_pipeline'

# Set the bulk size. If your indexing requests are timing out, make this
# smaller.
BULK_SIZE = 1000
NUMBER_OF_MOVIES = 100000
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
FAISS_SQ_FIELD = {
  "embedding": {
    "type": "knn_vector",
    "dimension": model_utils.DENSE_MODELS_HF[MODEL_SHORT_NAME]['dimensions'],
    "space_type": "l2",
    "method": {
      "name": "hnsw",
      "engine": "faiss",
      "parameters": {
        "encoder": {
          "name": "sq",
          "parameters": {
            "clip": True
          }
        },
        "m": 64,
        "ef_construction": 512
      }
    }
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


simple_ann_query={
  "query": {
    "knn": {
      EMBEDDING_FIELD_NAME: {
        "vector": [],
        "k": 10
      }
    }
  }
}


# Main function. Finds or loads the embedding model, creates the index (unless
# --skip-indexing is a command-line paramater), creates an embedding for the
# query "Sci-fi about the force and jedis" and then runs the exact query and
# prints the search response.
def main(skip_indexing=False, user_query=None):
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
  #
  # NOTE: Indexing takes an hour or more, depending on where you have deployed
  # the model
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
      pipeline_name=PIPELINE_NAME,
      additional_fields=FAISS_SQ_FIELD
    )

    # Read and add documents to the index with the opensearch-py bulk helper.
    logging.info(f"Indexing documents")
    counter = AutoIncrementingCounter()
    for bulk in movie_source.bulks(BULK_SIZE, INDEX_NAME):
      logging.info(f"Indexing bulk {str(counter)} / {TOTAL_NUMBER_OF_BULKS}")
      opensearchpy.helpers.bulk(os_client, bulk, timeout=600, max_retries=10)
  else:
    logging.info(f"Skipping indexing")

  # Run a query. Calls the LLM to generate a vector embedding for the question
  # (see model_utils.py) and then adds that embedding to the OpenSearch query.
  logging.info(f"Running query")
  query = deepcopy(simple_ann_query)
  query_embedding = model_utils.create_embedding(os_client, model_id, user_query)

  expr = jsonpath_ng.ext.parser.parse(f'query.knn.{EMBEDDING_FIELD_NAME}.vector')
  query = expr.update(query, query_embedding)
  response = os_client.search(index=INDEX_NAME, body=query, size=10)

  # Print the search response. The response contains the top 4 hits (the query
  # specifies "size": 4), which are the movies that are most similar to the
  # query.
  logging.info(f"Query response")
  for hit in response['hits']['hits']:
    logging.info(f"score: {hit['_score']}")
    logging.info(f"title: {hit['_source']['title']}")
    logging.info(f"plot: {hit['_source']['plot']}\n")
    logging.info(f"embedding source: {hit['_source'][EMBEDDING_SOURCE_FIELD_NAME]}")


if __name__ == "__main__":
  # Info level logging.
  logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

  parser = argparse.ArgumentParser(
      prog="main",
      description="Loads movie data, and runs approximate kNN queries. Use --skip-indexing"
      " to skip the from-scratch creation of the index.",
  )
  parser.add_argument("--skip-indexing", default=False, action="store_true")
  parser.add_argument("--query", default="Sci-fi about the force and jedis",
                      action="store")
  args = parser.parse_args()
  main(skip_indexing=args.skip_indexing,
       user_query=args.query)
