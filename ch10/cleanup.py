"""
A cleanup utility script for OpenSearch models and indices used in vector
similarity search examples.

This module provides functionality to clean up resources created by the vector
similarity search examples, including deployed ML models and indices. 

Command-line Arguments:
    --models: Flag to delete ML models
    --indices: Flag to delete indices
    
Example Usage:
    # Delete both models and indices python clean_up.py --models --indices
    
    # Delete only indices python clean_up.py --indices
    
    # Delete only models python clean_up.py --models

Notes:
    - Models are undeployed before deletion to prevent resource conflicts
    - Training indices are deleted after their target indices
"""
import argparse
import connector_utils
from copy import deepcopy
import opensearchpy
from os_client_factory import OSClientFactory
import logging
import model_utils
import time


def delete_models(os_client: opensearchpy.OpenSearch):
  all_dense = [model['name'] for model in model_utils.DENSE_MODELS_HF.values()]
  all_sparse = [model['name'] for model in model_utils.SPARSE_MODELS_HF.values()]
  for name in all_dense + all_sparse:
    model_id = model_utils.model_id_for(os_client=os_client,
                                        model_name=name)
    if model_id:
      logging.info(f'Deleting model {name}: {model_id}')
      os_client.transport.perform_request('POST', f'/_plugins/_ml/models/{model_id}/_undeploy')
      # This sleep prevents overwhelming the cluster with too many tasks and
      # responding with HTTP status 429
      time.sleep(1)
      os_client.transport.perform_request('DELETE', f'/_plugins/_ml/models/{model_id}')


# BEFORE running clean_up --indices, make sure these names match!
APPROXIMATE_FAISS_SQ = 'approximate_movies_sq'
APPROXIMATE_HNSW = 'approximate_movies_hnsw'
APPROXIMATE_IVF = 'approximate_movies_ivf'
APPROXIMATE_IVF_PQ = 'approximate_movies_ivf_pq'
EXACT = 'exact_movies'
IVF_TRAINING = 'ivf_training'
IVF_PQ_TRAINING = 'ivf_pq_training'
IVF_TRAINING_MODEL_NAME = 'ivf_model'
IVF_PQ_TRAINING_MODEL_NAME = 'ivf_pq_model'

def delete_indices(os_client: opensearchpy.OpenSearch):
  logging.warning("This script uses hard-coded index and training model names"
                  "If you have made any changes to these index names, also"
                  "update the clean_up script to match them.")
  for index_name in [APPROXIMATE_FAISS_SQ,
                     APPROXIMATE_HNSW,
                     APPROXIMATE_IVF,
                     APPROXIMATE_IVF_PQ,
                     EXACT,
                     # Important to delete these after their target indices!
                     IVF_TRAINING,
                     IVF_PQ_TRAINING,
                     ]:
    if os_client.indices.exists(index=index_name):
      logging.info(f'Deleting index {index_name}')
      os_client.indices.delete(index=index_name)
      time.sleep(1)
  try:
    os_client.transport.perform_request(
    'DELETE', f'/_plugins/_knn/models/{IVF_TRAINING_MODEL_NAME}')
  except opensearchpy.exceptions.NotFoundError:
    logging.info(f'Model {IVF_TRAINING_MODEL_NAME} not found, skipping')

  try:
    os_client.transport.perform_request(
    'DELETE', f'/_plugins/_knn/models/{IVF_PQ_TRAINING_MODEL_NAME}')
  except opensearchpy.exceptions.NotFoundError:
    logging.info(f'Model {IVF_PQ_TRAINING_MODEL_NAME} not found, skipping')


def delete_connectors(os_client: opensearchpy.OpenSearch):
  connector_id = connector_utils.connector_id_for(os_client=os_client,
                                                  connector_name='Amazon Bedrock')
  while connector_id:
    logging.info(f'Deleting model for connector {connector_id}')
    model_id = connector_utils.connector_model_id(os_client=os_client,
                                                  connector_id=connector_id)
    logging.info(f'Model id "{model_id}"')
    os_client.transport.perform_request('POST', f'/_plugins/_ml/models/{model_id}/_undeploy')
    # This sleep prevents overwhelming the cluster with too many tasks and
    # responding with HTTP status 429
    time.sleep(1)
    os_client.transport.perform_request('DELETE', f'/_plugins/_ml/models/{model_id}')

    logging.info(f'Deleting connector {connector_id}')
    os_client.transport.perform_request('DELETE', f'/_plugins/_ml/connectors/{connector_id}')
    connector_id = connector_utils.connector_id_for(os_client=os_client,
                                                    connector_name='Amazon Bedrock')


def main(clean_models=False, clean_indices=False, clean_connectors=False):
  os_client = OSClientFactory().client()
  if clean_models:
    logging.info("Deleting models")
    delete_models(os_client)
  if clean_indices:
    logging.info("Deleting indices")
    delete_indices(os_client)
  if clean_connectors:
    logging.info("Deleting connectors")
    delete_connectors(os_client)
  logging.info("Done!")


if __name__ == "__main__":
  # Info level logging.
  logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

  parser = argparse.ArgumentParser(
      prog="main",
      description="Cleans up models and indices.",
  )
  parser.add_argument("--models", default=False, action="store_true")
  parser.add_argument("--indices", default=False, action="store_true")
  parser.add_argument("--connectors", default=False, action="store_true")

  args = parser.parse_args()
  main(clean_models=args.models,
       clean_indices=args.indices,
       clean_connectors=args.connectors)
