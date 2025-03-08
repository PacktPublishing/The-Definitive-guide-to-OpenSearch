"""
A module for training an IVF-PQ (Inverted File with Product Quantization) model
in OpenSearch.

This module handles the training process for IVF-PQ approximate k-NN search,
which requires a separate training step before creating the search index. It
loads sample movie data, generates vector embeddings, and trains the IVF-PQ
model using those embeddings.

Key Features:
    - Creates a training index for vector data
    - Sets up an ingest pipeline for embedding generation
    - Indexes a subset of documents for training (10% of total documents)
    - Trains an IVF-PQ model with specified parameters
    - Monitors training completion status

Note:
    - Training requires approximately 10% of the total document set
    - The training process must complete before creating the IVF-PQ search index
    - Model training parameters (m=8, code_size=8) are configured for general
      use
"""
from copy import deepcopy
import index_utils
import logging
import movie_source
from opensearchpy import OpenSearch
import opensearchpy.helpers
import time


# We recommend 10% of the total documents for training.
MOVIES_TO_TRAIN = int(movie_source.TOTAL_MOVIES * .10)
DOCS_PER_BULK = MOVIES_TO_TRAIN // 10
TOTAL_NUMBER_OF_BULKS = int(MOVIES_TO_TRAIN / DOCS_PER_BULK)


# Defines the training index name. The script loads raw vector data into this
# index and calls the train API to prepare for creating the IVF PQ index
TRAINING_INDEX_NAME = 'ivf_pq_training'
TRAINING_MODEL_NAME = 'ivf_pq_model'
TRAINING_SOURCE_FIELD_NAME = 'embedding_source'
TRAINING_DEST_FIELD_NAME = 'embedding'
TRAINING_PIPELINE_NAME = 'ivf_pq_pipeline'
from approximate_ivf_pq import INDEX_NAME as DESTINATION_INDEX_NAME


# Defines the index mapping for training data with basic movie metadata fields.
# Note: This mapping deliberately excludes knn:true setting in the index settings
# since you use raw vector data for training the IVF model.
# The embedding vector field is added dynamically during index creation
TRAINING_INDEX_MAPPING = {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 1
      # Do not set knn: true for the training index!
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
}}}}}


# Ingest pipeline that the code uses to create the vector embeddings it loads
# into the training index
training_ingest_pipeline_definition = {
  "description": "Embedding pipeline for training data",
  "processors": [
    {
      "text_embedding": {
        "model_id": "",
        "field_map": {
          TRAINING_SOURCE_FIELD_NAME: TRAINING_DEST_FIELD_NAME
        }
      }
    }
  ]
}


# The request body for the train API call. This body specifies the engine and
# method parameters and the algorithm parameters that will be used in the target
# index for storing and searching
TRAINING_REQUEST_BODY = {
  "training_index": TRAINING_INDEX_NAME,
  "training_field": TRAINING_DEST_FIELD_NAME,
  "description": "Training model for IVF PQ",
  "space_type": "l2",
  "method": {
    "name": "ivf",
    "engine": "faiss",
    "parameters": {
      "encoder": {
        "name": "pq",
        "parameters": {
          "m": 8,
          "code_size": 8
}}}}}


# Retrieves the current state of the IVF training model from OpenSearch.
# Returns:
#     str or None: Current state of the model if it exists, None if the model
#                 is not found
def _get_model_state(os_client):
  try:
    model_response = os_client.transport.perform_request(
      'GET', f'/_plugins/_knn/models/{TRAINING_MODEL_NAME}?filter_path=state&pretty',
    )
    return model_response['state']
  except opensearchpy.exceptions.NotFoundError as e:
    return None


# Polls the IVF model training state until completion or timeout. This helper
# function continuously monitors the training progress by checking the model
# state at regular intervals. It will wait until the model leaves the training
# state
def _wait_for_training_completion(os_client, model_id, model_dimensions):
  logging.info(f"Waiting for training to complete for model {TRAINING_MODEL_NAME}")
  state = _get_model_state(os_client)
  while state and state == "training":
    state = _get_model_state(os_client)
    time.sleep(1)


# Main entry point. Call train to do the PQ training on 10% of the source data,
# preparing for indexing the full corpus.
def train(os_client: OpenSearch, embedding_model_id, model_dimensions, skip_if_exists=True):

  # If the model already exists, and skip_if_exists is true, then don't create a
  # new model. Otherwise, delete the existing model.
  state = _get_model_state(os_client)
  if state and state == 'created':
    logging.info(f"Model {TRAINING_MODEL_NAME} already exists.")
    if skip_if_exists:
      logging.info(f"Skipping training for {TRAINING_MODEL_NAME}")
      return
    else:
      logging.info(f"Deleting model {TRAINING_MODEL_NAME}")
      if os_client.indices.exists(index=DESTINATION_INDEX_NAME):
        logging.info(f"Index {DESTINATION_INDEX_NAME} is using the model, deleting that index")
        os_client.indices.delete(index=DESTINATION_INDEX_NAME)
      os_client.transport.perform_request(
        'DELETE', f'/_plugins/_knn/models/{TRAINING_MODEL_NAME}'
      )

  # Create an ingest pipeline
  logging.info(f"Creating ingest pipeline {TRAINING_PIPELINE_NAME}")
  pipeline_definition = deepcopy(training_ingest_pipeline_definition)
  pipeline_definition['processors'][0]['text_embedding']['model_id'] = embedding_model_id
  os_client.ingest.put_pipeline(id=TRAINING_PIPELINE_NAME, body=pipeline_definition)

  # Create the training index
  logging.info(f"Creating training index {TRAINING_INDEX_NAME}")
  index_utils.delete_then_create_index(os_client=os_client,
                                        index_name=TRAINING_INDEX_NAME,
                                        ingest_pipeline_name=TRAINING_PIPELINE_NAME,
                                        additional_fields={
                                          TRAINING_DEST_FIELD_NAME: {
                                            "type": "knn_vector",
                                            "dimension": model_dimensions
                                        }})

  # Index documents to the training index.
  logging.info(f"Indexing documents for training")
  bulks_sent = 0
  for bulk in movie_source.bulks(DOCS_PER_BULK, TRAINING_INDEX_NAME):
    logging.info(f"Indexing bulk {bulks_sent + 1} / {TOTAL_NUMBER_OF_BULKS}")
    opensearchpy.helpers.bulk(os_client, bulk, timeout=600, max_retries=10)
    bulks_sent += 1
    if bulks_sent >= TOTAL_NUMBER_OF_BULKS:
      break

  # Train the model
  logging.info(f"Sending train request for {TRAINING_MODEL_NAME}")
  training_request_body = deepcopy(TRAINING_REQUEST_BODY)
  training_request_body['dimension'] = model_dimensions
  os_client.transport.perform_request(
    'POST', f'/_plugins/_knn/models/{TRAINING_MODEL_NAME}/_train',
    body=training_request_body
  )
  logging.info(f"Waiting for training to complete for model {TRAINING_MODEL_NAME}")
  _wait_for_training_completion(os_client, embedding_model_id, model_dimensions)
  return TRAINING_MODEL_NAME
