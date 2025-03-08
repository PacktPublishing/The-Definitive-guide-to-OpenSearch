import copy
from opensearchpy import OpenSearch
import logging
import time


CONNECTOR_NAME = 'Amazon Bedrock'
CONNECTOR_REGISTER_BODY = {
    "name": '',
    "function_name": 'remote',
    'description': 'bedrock',
    'connector_id': ''
}


def _deploy_connector(os_client: OpenSearch, body):
  response = os_client.transport.perform_request(
    'POST', '/_plugins/_ml/connectors/_create',
    body=body)
  logging.info(f"_deploy_connector, connector_id {response['connector_id']}")
  return response['connector_id']
  

def _register_connector(os_client: OpenSearch, body):
  response = os_client.transport.perform_request(
    'POST', '/_plugins/_ml/models/_register',
    body=body)
  return response['task_id']


def _wait_deploy_connector(os_client: OpenSearch, task_id):
  response = os_client.transport.perform_request('GET', f'/_plugins/_ml/tasks/{task_id}')
  logging.info(f"Task status: {response['state']}")
  state = response['state']
  while state != 'COMPLETED':
    if state == 'FAILED' or state == 'COMPLETED_WITH_ERROR':
      raise Exception(f"Task failed: {response}")
    response = os_client.transport.perform_request('GET', f'/_plugins/_ml/tasks/{task_id}')
    state = response['state']
    logging.info(f"Task status: {state}")
    # This needs a delay, or it will overwhelm OpenSearch with these
    # calls and eventually OpenSearch will return a HTTP 429 status. 
    time.sleep(1)
  return response


def connector_id_for(os_client: OpenSearch, connector_name):
  logging.info(f"Searching for connector {connector_name}")
  connectors = os_client.transport.perform_request(
    'GET', '/_plugins/_ml/connectors/_search',
    body={
      "size": 10000,
      "query": {"match_all": {}}
    }
  )
  for connector in connectors['hits']['hits']:
    logging.info(f"Connector: {connector['_source']['name']}")
    if connector['_source']['name'] == connector_name:
      return connector['_id']
  return None


def connector_model_id_for_connector(os_client: OpenSearch, connector_id):
  response = os_client.transport.perform_request(
    'GET', f'/_plugins/_ml/models/_search', body={"size": 10000})
  for model in response['hits']['hits']:
    retrieved_connector_id = model['_source'].get('connector_id', None)
    if retrieved_connector_id == connector_id:
      return model['_id']
  return None


def delete_then_create_connector(os_client: OpenSearch, connector_name, connector_body):
  connector_id = connector_id_for(os_client=os_client, connector_name=connector_name)
  if connector_id is not None:
    logging.info(f"Connector {connector_name} found. Deleting...")
    model_id = connector_model_id_for_connector(os_client=os_client,
                                  connector_id=connector_id)
    logging.info(f'Found model_id for connector {connector_name}: "{model_id}". Deleting it.')
    os_client.transport.perform_request('POST', f'/_plugins/_ml/models/{model_id}/_undeploy')
    time.sleep(1)
    os_client.transport.perform_request('DELETE', f'/_plugins/_ml/models/{model_id}')
    logging.info(f'Deleting connector {connector_id}')
    os_client.transport.perform_request('DELETE', f'/_plugins/_ml/connectors/{connector_id}')

  connector_id = _deploy_connector(os_client=os_client, body=connector_body)
  logging.info(f'Connector ID after deploy {connector_id}')

  register_body = copy.deepcopy(CONNECTOR_REGISTER_BODY)
  register_body['name'] = connector_name
  register_body['connector_id'] = connector_id
  task_id = _register_connector(os_client=os_client, body=register_body)
  response = _wait_deploy_connector(os_client=os_client, task_id=task_id)
  return {'connector_id': connector_id,
          'model_id': response['model_id']}


if __name__ == '__main__':
  import os_client_factory
  import boto3
  logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

  session = boto3.client('sts', os_client_factory.AWS_REGION).get_session_token()
  os_client = os_client_factory.OSClientFactory().client()
  delete_then_create_connector(os_client=os_client,
                           connector_name=CONNECTOR_NAME,
                           connector_body={
    "name": "Amazon Bedrock",
    "description": "Test connector for Amazon Bedrock",
    "version": 1,
    "protocol": "aws_sigv4",
    "credential": {
        "access_key": session['Credentials']['AccessKeyId'],
        "secret_key": session['Credentials']['SecretAccessKey'],
        "session_token": session['Credentials']['SessionToken']
    },
    "parameters": {
        "region": f"{os_client_factory.AWS_REGION}",
        "service_name": "bedrock",
        "model": "anthropic.claude-v2"
    },
    "actions": [
        {
            "action_type": "predict",
            "method": "POST",
            "headers": {
                "content-type": "application/json"
            },
            "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/invoke",
            "request_body": "{\"prompt\":\"\\n\\nHuman: ${parameters.inputs}\\n\\nAssistant:\",\"max_tokens_to_sample\":300,\"temperature\":0.5,\"top_k\":250,\"top_p\":1,\"stop_sequences\":[\"\\\\n\\\\nHuman:\"]}"
        }
    ]
})