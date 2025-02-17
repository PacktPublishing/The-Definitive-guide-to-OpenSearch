'''
Use with caution!

This script deletes all models in OpenSearch that match the name of one of the
hugging face models from model_utils.py. This can be useful if you've deployed
many models and run out of RAM
'''
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
    time.sleep(1)
    os_client.transport.perform_request('DELETE', f'/_plugins/_ml/models/{model_id}')
