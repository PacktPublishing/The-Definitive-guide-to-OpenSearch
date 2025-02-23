'''
Utility functions to help manage models on OpenSearch

Call find_or_deploy_model with the model name, and the model 
definition for the _ml plugin's register API. If a model by the
given name is already deployed, returns that model's ID.
Otherwise, calls the register API and blocks for the registration
and deployment of the model.
'''
from opensearchpy import OpenSearch
import logging
import time


# Models and dimensions deployable directly to OpenSearch.
HUGGING_FACE_MODELS = {
  "all-distilroberta-v1": {
    "name": "huggingface/sentence-transformers/all-distilroberta-v1",
    "dimensions": 768,
    "version": "1.0.1"
  },
  "all-MiniLM-L6-v2": {
    "name": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
    "dimensions": 384,
    "version": "1.0.1"
  },
  "all-MiniLM-L12-v2": {
    "name": "huggingface/sentence-transformers/all-MiniLM-L12-v2",
    "dimensions": 384,
    "version": "1.0.1"
  },
  "all-mpnet-base-v2": {
    "name": "huggingface/sentence-transformers/all-mpnet-base-v2",
    "dimensions": 768,
    "version": "1.0.1"
  },
  "msmarco-distilbert-base-tas-b": {
    "name": "huggingface/sentence-transformers/msmarco-distilbert-base-tas-b",
    "dimensions": 768,
    "version": "1.0.2"
  },
  "multi-qa-MiniLM-L6-cos-v1": {
    "name": "huggingface/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "dimensions": 384,
    "version": "1.0.1"
  },
  "multi-qa-mpnet-base-dot-v1": {
    "name": "huggingface/sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "dimensions": 384,
    "version": "1.0.1"
  },
  "paraphrase-MiniLM-L3-v2": {
    "name": "huggingface/sentence-transformers/paraphrase-MiniLM-L3-v2",
    "dimensions": 384,
    "version": "1.0.1"
  },
  "paraphrase-multilingual-MiniLM-L12-v2": {
    "name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "dimensions": 384,
    "version": "1.0.1"
  },
  "paraphrase-mpnet-base-v2": {
    "name": "huggingface/sentence-transformers/paraphrase-mpnet-base-v2",
    "dimensions": 768,
    "version": "1.0.0"
  },
  "distiluse-base-multilingual-cased-v1": {
    "name": "huggingface/sentence-transformers/distiluse-base-multilingual-cased-v1",
    "dimensions": 512,
    "version": "1.0.1"
  },
}


# Calls the _ml plugin's tasks API with the given task, and returns
# when the task is marked "COMPLETED". If the task fails or has an
# error condition, raises an execption instead.
#
# Returns the http response for the final call to the tasks API.
def _monitor_task(os_client: OpenSearch, task_id):
  response = os_client.transport.perform_request('GET', f'/_plugins/_ml/tasks/{task_id}')
  logging.info(f"{int(time.time())}: Task status: {response['state']}")
  state = response['state']
  while state != 'COMPLETED':
    if state == 'FAILED' or state == 'COMPLETED_WITH_ERROR':
      raise Exception(f"{int(time.time())}: Task failed: {response}")
    response = os_client.transport.perform_request('GET', f'/_plugins/_ml/tasks/{task_id}')
    state = response['state']
    logging.info(f"{int(time.time())}: Task status: {state}")
    # This needs a delay, or it will overwhelm OpenSearch with these
    # calls and eventually OpenSearch will return a HTTP 429 status. 
    time.sleep(1)
  return response


# Call after making the register_model API call, with the task_id 
# from the response. This calls monitor_task to busy-wait for its
# completion. Once the model is registered, this calls the API
# to deploy the model and busy-waits for the deployment to complete.
#
# Returns the model_id for the deployed model.
def _deploy_and_get_model_id(os_client: OpenSearch, task_id):
  model_id = _monitor_task(os_client=os_client, 
                           task_id=task_id)["model_id"]
  # Seems like this is necessary
  response = os_client.transport.perform_request(
    'POST', f'/_plugins/_ml/models/{model_id}/_deploy'
  )
  _monitor_task(os_client=os_client, task_id=response['task_id'])
  return model_id


# Calls OpenSearch ml plugin's models API to get the model_id for
# the given model_name. 
# 
# Returns the model id, or None if the model is not found.
def model_id_for(os_client: OpenSearch, model_name):
  models = os_client.transport.perform_request(
    'GET', '/_plugins/_ml/models/_search',
    body={
      "size": 10000,
      "query": {"match_all": {}}
    }
  )
  for model in models['hits']['hits']:
    if model['_source']['name'] == model_name:
      model_id = model['_source'].get('model_id', None)
      if model_id:
        return model_id
  return None


# Calls the OpenSearch ml plugin's register_model API to register
# the model. This is a blocking call that returns when the model
# is registered.
#
# Specify the JSON `body` for the register_model API call.
#
# Returns the model_id for the registered model.
def find_or_deploy_model(os_client: OpenSearch, model_name, body):
  model_id = model_id_for(os_client=os_client,
                          model_name=model_name)
  if model_id is None:
    logging.info(f"Model {model_name} not found. Deploying...")
    response = os_client.transport.perform_request(
      'POST', '/_plugins/_ml/models/_register',
      body=body
    )
    task_id = response['task_id']
    model_id = _deploy_and_get_model_id(os_client=os_client, task_id=task_id)
  else:
    logging.info(f"Model {model_name} found. Skipping deployment.")
  return model_id


# Use this to call the _predict API for a loaded, dense model.
#
# Returns the vector for the text.
def create_embedding(os_client, model_id, input_text):
  response = os_client.transport.perform_request(
    'POST', f'/_plugins/_ml/_predict/text_embedding/{model_id}',
    body={
      "text_docs": [input_text],
      "return_number": True,
      "target_response": ["sentence_embedding"]
    }
  )
  return response['inference_results'][0]['output'][0]['data']
