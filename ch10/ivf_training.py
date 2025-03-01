"""
IVF (Inverted File Index) training module for OpenSearch vector search.

This module handles the training process for IVF-based approximate k-NN search
in OpenSearch. It manages the creation of training data, index setup, and model
training for vector similarity search.

Key Components:
    - Training index creation with appropriate mappings
    - Ingest pipeline setup for text embedding
    - IVF model training configuration and execution
    - Model state monitoring

Constants:
    DOCS_PER_BULK (int): Number of documents to process in each bulk operation
    TOTAL_NUMBER_OF_BULKS (int): Total number of bulk operations to perform
    TRAINING_INDEX_NAME (str): Name of the index used for training
    TRAINING_MODEL_NAME (str): Name of the IVF model to be trained
    TRAINING_SOURCE_FIELD_NAME (str): Source field for text embedding
    TRAINING_DEST_FIELD_NAME (str): Destination field for embedded vectors
    TRAINING_PIPELINE_NAME (str): Name of the ingest pipeline

Functions:
    train: Main function to execute the IVF training process
    _get_model_state: Helper function to check training model state
    _wait_for_training_completion: Helper function to monitor training progress

Dependencies:
    - opensearchpy
    - index_utils
    - model_utils
    - movie_source

Note:
    The training process requires a text embedding model to be deployed in
    OpenSearch before execution.
"""


from copy import deepcopy
import index_utils
import logging
import model_utils
import movie_source
from opensearchpy import OpenSearch
import opensearchpy.helpers
import time


# We recommend 10% of the total documents for training.
BULK_SIZE = 100
DOCS_PER_BULK = 1000
TOTAL_NUMBER_OF_BULKS = 10


# Defines the training index name. The script loads raw vector data into this
# index and calls the train API to prepare for creating the IVF index
TRAINING_INDEX_NAME = 'approximate_movies_training'
TRAINING_MODEL_NAME = 'approximate_movies_model'
TRAINING_SOURCE_FIELD_NAME = 'embedding_source'
TRAINING_DEST_FIELD_NAME = 'embedding'
TRAINING_PIPELINE_NAME = 'training_pipeline'


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
                   }},
    }}}


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
  "description": "Training model for IVF",
  "space_type": "l2",
  "method": {
    "name": "ivf",
    "engine": "faiss",
    "parameters": {
      "nlist": 4,
      "nprobes": 2
}}}


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



"""
Executes the IVF model training process for vector similarity search.

This function performs the complete training workflow:
1. Checks if model already exists and handles according to skip_if_exists
2. Creates an ingest pipeline for text embedding
3. Creates a training index with appropriate mappings
4. Indexes training documents (10% of total dataset)
5. Initiates IVF model training
6. Monitors training completion

Args:
    os_client (OpenSearch): OpenSearch client instance for API operations
    model_id (str): ID of the text embedding model to use for vector creation
    model_dimensions (int): Dimension size of the embedding vectors
    skip_if_exists (bool, optional): If True, skips training when model exists.
                                    If False, deletes and retrains. Defaults to True.

Returns:
    str: Name of the trained model (TRAINING_MODEL_NAME)

Note:
    - Training requires approximately 10% of total documents for optimal results
    - The process can take several minutes depending on data size
    - Existing models will be preserved if skip_if_exists=True
"""
def train(os_client: OpenSearch, model_id, model_dimensions, skip_if_exists=True):

  # If the model already exists, and skip_if_exists is true, then don't create a
  # new model. Otherwise, delete the existing model.
  logging.info(f'Model state: _get_model_state(os_client)')
  state = _get_model_state(os_client)
  if state and state == 'created':
    logging.info(f"Model {TRAINING_MODEL_NAME} already exists.")
    if skip_if_exists:
      logging.info(f"Skipping training for {TRAINING_MODEL_NAME}")
      return
    else:
      logging.info(f"Deleting model {TRAINING_MODEL_NAME}")
      os_client.transport.perform_request(
        'DELETE', f'/_plugins/_knn/models/{TRAINING_MODEL_NAME}'
      )

  # Create an ingest pipeline
  logging.info(f"Creating ingest pipeline {TRAINING_PIPELINE_NAME}")
  pipeline_definition = deepcopy(training_ingest_pipeline_definition)
  pipeline_definition['processors'][0]['text_embedding']['model_id'] = model_id
  os_client.ingest.put_pipeline(id=TRAINING_PIPELINE_NAME, body=pipeline_definition)

  # Create the training index
  logging.info(f"Creating training index {TRAINING_INDEX_NAME}")
  index_utils.delete_then_create_index(os_client=os_client,
                                        index_name=TRAINING_INDEX_NAME,
                                        pipeline_name=TRAINING_PIPELINE_NAME,
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
  _wait_for_training_completion(os_client, model_id, model_dimensions)
  return TRAINING_MODEL_NAME
