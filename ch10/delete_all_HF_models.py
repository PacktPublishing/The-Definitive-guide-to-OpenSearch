"""
Model Cleanup Utility

This script provides functionality to clean up deployed Hugging Face models from OpenSearch.
It iterates through the models defined in model_utils.py and removes their deployments
to free up RAM and resources.

The script will:
1. Connect to OpenSearch using the client factory
2. Find all deployed models matching Hugging Face model names
3. Undeploy each model
4. Delete the model definition
5. Add delays between operations to prevent overwhelming the cluster

Usage:
    Run this script directly to remove all deployed Hugging Face models.
    
Warning:
    Use with caution! This will permanently delete model deployments.
    Make sure you want to remove all models before running.

Dependencies:
    - os_client_factory
    - model_utils
    - logging
    - time
"""


from os_client_factory import OSClientFactory
import logging
import model_utils
import time


logging.basicConfig(level=logging.INFO)

os_client = OSClientFactory().client()
for name in model_utils.HUGGING_FACE_MODELS.keys():
  model_full_name = model_utils.HUGGING_FACE_MODELS[name]['name']
  model_id = model_utils.model_id_for(os_client=os_client,
                                      model_name=model_full_name)
  if model_id:
    logging.info(f'Deleting model {name}: {model_id}')
    os_client.transport.perform_request('POST', f'/_plugins/_ml/models/{model_id}/_undeploy')
    # This sleep prevents overwhelming the cluster with too many tasks and
    # responding with HTTP status 429
    time.sleep(1)
    os_client.transport.perform_request('DELETE', f'/_plugins/_ml/models/{model_id}')
