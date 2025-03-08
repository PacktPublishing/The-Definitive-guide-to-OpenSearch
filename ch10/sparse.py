import argparse
from auto_incrementing_counter import AutoIncrementingCounter
from copy import deepcopy
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
INDEX_NAME = 'sparse_movies'
PIPELINE_NAME = 'sparse_pipeline'

# Set the bulk size. If your indexing requests are timing out, make this
# smaller.
BULK_SIZE = 1000
NUMBER_OF_MOVIES = movie_source.TOTAL_MOVIES
TOTAL_NUMBER_OF_BULKS = NUMBER_OF_MOVIES // BULK_SIZE

# This is the sparse vector generating model. Used for both bi_encoder and
# doc_only sparse vector generation during ingest
MODEL_SHORT_NAME = "opensearch-neural-sparse-encoding-v2-distill"
MODEL_REGISTER_BODY = {
  "name": model_utils.SPARSE_MODELS_HF[MODEL_SHORT_NAME]['name'],
  "model_format": "TORCH_SCRIPT",
  "version": model_utils.SPARSE_MODELS_HF[MODEL_SHORT_NAME]['version']
}


# This is the tokenizer model. Used for tokenizing the query and the
# document during sparse vector generation.
TOKENIZER_SHORT_NAME = "opensearch-neural-sparse-tokenizer-v1"
TOKENIZER_REGISTER_BODY = {
  "name": model_utils.SPARSE_MODELS_HF[TOKENIZER_SHORT_NAME]['name'],
  "model_format": "TORCH_SCRIPT",
  "version": model_utils.SPARSE_MODELS_HF[TOKENIZER_SHORT_NAME]['version']
}


# Defines the source text for the embedding field. You can modify
# movie_source.py to change how the data is treated. 
EMBEDDING_SOURCE_FIELD_NAME = 'embedding_source'
EMBEDDING_FIELD_NAME = 'embedding'
KNN_FIELDS = {
  "embedding": {
    "type": "rank_features"
  }
}


# Definition for the ingest pipeline. Maps the EMBEDDING_SOURCE_FIELD to the
# EMBEDDING_FIELD. OpenSearch neural plugin uses these fields for creating the
# embedding as you ingest data.
ingest_pipeline_definition = {
  "description": "Sparse encoding ingest pipeline",
  "processors": [
    {
      "sparse_encoding": {
        "model_id": "",
        "prune_type": "max_ratio",
        "prune_ratio": 0.1,
        "field_map": {
          EMBEDDING_SOURCE_FIELD_NAME: EMBEDDING_FIELD_NAME
        }
      }
    }
  ]
}


sparse_query = {
  "query": {
    "neural_sparse": {
      EMBEDDING_FIELD_NAME: {
        "query_text": "",
        # This model id can be either the encoder model or the tokenizer model
        "model_id": ""
      }
    }
  }
}


# Main function. Finds or loads the embedding model, creates the index (unless
# --skip-indexing is a command-line paramater), creates an embedding for the
# query and then runs the exact query and prints the search response.
def main(skip_indexing=False, bi_encoder=False, doc_only=False, user_query=None):
  logging.info(f"Query: {user_query}")

  # See os_client_factory.py for details on the set up for the opensearch-py
  # client.
  os_client = OSClientFactory().client()

  # Find or register the sparse generation model.
  logging.info(f"Finding or deploying model {MODEL_SHORT_NAME}")
  model_id = model_utils.find_or_deploy_model(
    os_client=os_client,
    model_name=model_utils.SPARSE_MODELS_HF[MODEL_SHORT_NAME]['name'],
    body=MODEL_REGISTER_BODY
  )

  # Find or deploy the tokenizer model.
  logging.info(f"Finding or deploying model {TOKENIZER_SHORT_NAME}")
  tokenizer_id = model_utils.find_or_deploy_model(
    os_client=os_client,
    model_name=model_utils.SPARSE_MODELS_HF[TOKENIZER_SHORT_NAME]['name'],
    body=TOKENIZER_REGISTER_BODY
  )

  # If you did not disable indexing, this will create a new index, set up an
  # ingest pipeline for automatically generating vector embeddings on ingest,
  # read the movies data (movie_source.py) and send it to the index.
  if not skip_indexing:

    # Create an ingest pipeline
    pipeline_definition = deepcopy(ingest_pipeline_definition)
    pipeline_definition['processors'][0]['sparse_encoding']['model_id'] = model_id
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

  # Run a query. 
  logging.info(f"Running query")
  if doc_only:
    query = deepcopy(sparse_query)
    query['query']['neural_sparse'][EMBEDDING_FIELD_NAME]['model_id'] = tokenizer_id
  elif bi_encoder:
    query = deepcopy(sparse_query)
    query['query']['neural_sparse'][EMBEDDING_FIELD_NAME]['model_id'] = model_id
  query['query']['neural_sparse'][EMBEDDING_FIELD_NAME]['query_text'] = \
    user_query if user_query else "Sci-fi about the force and jedis"

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
  parser.add_argument("--bi-encoder", default=False, action="store_true")
  parser.add_argument("--doc-only", default=False, action="store_true")
  parser.add_argument("--query", default="Sci-fi about the force and jedis",
                      action="store")
  args = parser.parse_args()
  
  if (not args.bi_encoder and not args.doc_only) or \
     (args.bi_encoder and args.doc_only):
    logging.info("Specify exactly one of --bi-encoder, or --doc-only ")
    logging.info("Setting --bi-encoder by default")
    args.bi_encoder = True
    
  main(skip_indexing=args.skip_indexing,
       bi_encoder=args.bi_encoder,
       doc_only=args.doc_only,
       user_query=args.query)
