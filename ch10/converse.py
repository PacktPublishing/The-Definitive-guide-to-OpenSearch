import argparse
from auto_incrementing_counter import AutoIncrementingCounter
import boto3
import copy
import connector_utils
import index_utils
import logging
import movie_source
from os_client_factory import OSClientFactory, AWS_REGION
import opensearchpy.helpers
import uuid


# Defines the index created by the script.
INDEX_NAME = 'conversational_movies'
SEARCH_PIPELINE_NAME = 'rag_pipeline'


# Set the bulk size. If your indexing requests are timing out, make this
# smaller.
BULK_SIZE = 1000
NUMBER_OF_MOVIES = movie_source.TOTAL_MOVIES
TOTAL_NUMBER_OF_BULKS = NUMBER_OF_MOVIES // BULK_SIZE


# The connector body specifies credentials, and the model for Bedrock. It also
# encapsulates the prompt that goes to Bedrock. Refer to the project's
# ml_commons repo for connector blueprints to connect to other model hosts.
#
# https://github.com/opensearch-project/ml-commons/tree/2.x/docs/remote_inference_blueprints
#
# To use other hosts, you also need to modify the
# trusted_connector_endpoints_regex cluster setting in os_client_factory.py
CONNECTOR_BODY = {
  "name": connector_utils.CONNECTOR_NAME,
  "description": "Test connector for Amazon Bedrock",
  "version": 1,
  "protocol": "aws_sigv4",
  "credential": '',
  "interface": {},
  "parameters": {
    "region": f"{AWS_REGION}",
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
}


# The search pipeline calls out to a text generation model to create a response
# based on the search results for the query. 
SEARCH_PIPELINE_BODY={
  "response_processors": [
    {
      "retrieval_augmented_generation": {
        "tag": "conversation demo",
        "description": "Demo pipeline Using Bedrock Connector",
        "model_id": '', # This is the model id for the connector
        "context_field_list": ["plot", "title"],
        "system_prompt": "You are a helpful assistant",
        "user_instructions": "Generate a concise and informative answer in less than 300 "
                             "words for the given question, using information from the "
                             "search results. If the search results do not contain the "
                             "answer, respond with \"Sorry, I don't know.\". Do not "
                             "include any information that is not in the search results.",
      }
    }
  ]
}

# For this example, we chose a simple_query_string query, rather than a semantic
# or other query. You can switch the query to e.g., add filters, use semantic
# search, or use a hybrid query
CONVERSATION_SEARCH_QUERY = {
  "query": {
    "simple_query_string": {
      "query": '',
      "fields": ['title^2', 'plot']
    }
  },
  "_source": ["title", "plot", "embedding_source"],
  "ext": {
    "generative_qa_parameters": {
      "llm_model": "bedrock/claude",
      "llm_question": '',
      "memory_id": '',
      "context_size": 5,
      "message_size": 5,
      "timeout": 60
    }
  }
}


def create_search_pipeline(os_client, pipeline_name, pipeline_body):
  '''Creates the search pipeline'''
  response = os_client.transport.perform_request(
    'PUT', f'/_search/pipeline/{pipeline_name}',
    body=pipeline_body)


def create_conversation_memory(os_client):
  '''Creates a memory for the conversation'''
  conversation_name = f'conversation-{str(uuid.uuid1())[:8]}'
  response = os_client.transport.perform_request(
    'POST', '/_plugins/_ml/memory/',
    body={"name": conversation_name}
  )
  return response['memory_id']


def main(skip_indexing=False):
  '''Sets up and runs a conversational chat bot using OpenSearch and Amazon Bedrock.

    This function performs the following operations:
    1. Creates an OpenSearch client
    2. If skip_indexing is False:
        - Sets up an Amazon Bedrock connector with temporary credentials
        - Creates a search pipeline with retrieval augmented generation
        - Creates a conversation memory
        - Creates and populates an index with movie data
    3. Enters an interactive loop where users can ask questions about movies
    
    Args:
        skip_indexing (bool, optional): If True, skips the index creation and 
            data loading steps. Defaults to False.

    Note:
        Requires valid AWS credentials with Bedrock access configured for 
        Anthropic Claude in the specified AWS region.'''
  # See os_client_factory.py for details on the set up for the opensearch-py
  # client. 
  os_client = OSClientFactory().client()

  # The conversation memory is automatically maintained by the search
  # processor. The id is injected into the query processor before the query is
  # executed
  conversation_memory_id = create_conversation_memory(os_client)

  if not skip_indexing:
    # Set up the connector for Amazon Bedrock. This uses the default profile
    # for the AWS CLI. If you want to use a different profile, you can
    # specify it in the AWS_DEFAULT_PROFILE environment variable.
    #
    # IMPORTANT! You must have Bedrock model access configured to give you
    # access to Anthropic Claude in the AWS_REGION specified in
    # os_client_factory.py
    logging.info(f"Creating connector")
    # Secure token service (sts) provides temporary credentials based on the
    # account specified by aws configure, See the boto docs for details and
    # alternative ways to specify credentials.
    session = boto3.client('sts', AWS_REGION).get_session_token()
    connector_body = copy.deepcopy(CONNECTOR_BODY)
    connector_body['credential'] = {
      "access_key": session['Credentials']['AccessKeyId'],
      "secret_key": session['Credentials']['SecretAccessKey'],
      "session_token": session['Credentials']['SessionToken']
    }
    # This locates the connector by searching the connectors API for the
    # CONNECTOR_NAME, deletes any connectors and their associated model, then
    # creates a new connector with the body above
    #
    # The session token is embedded in the connector and expires after an hour.
    # By deleting any existing model, the code ensures that the session
    # credentials are active.
    deployed_connector = connector_utils.delete_then_create_connector(
      os_client=os_client,
      connector_name=connector_utils.CONNECTOR_NAME,
      connector_body=connector_body
    )

    # The retrieval_augmented_generation processor accesses the connector
    # through its associated model id. 
    model_id = deployed_connector['model_id']
    search_pipeline_body = copy.deepcopy(SEARCH_PIPELINE_BODY)
    search_pipeline_body['response_processors'][0]['retrieval_augmented_generation']['model_id'] = model_id
    create_search_pipeline(os_client, SEARCH_PIPELINE_NAME, search_pipeline_body)

    # Create the index. The search pipeline defined above is the default
    # pipeline for this index. All queries that go to the index will run this
    # pipeline. 
    logging.info(f"Creating index {INDEX_NAME}")
    index_utils.delete_then_create_index(
      os_client=os_client,
      index_name=INDEX_NAME,
      search_pipeline_name=SEARCH_PIPELINE_NAME
    )
    
    # Read and add documents to the index with the opensearch-py bulk helper.
    logging.info(f"Indexing documents")
    counter = AutoIncrementingCounter()
    for bulk in movie_source.bulks(BULK_SIZE, INDEX_NAME):
      logging.info(f"Indexing bulk {str(counter)} / {TOTAL_NUMBER_OF_BULKS}")
      opensearchpy.helpers.bulk(os_client, bulk, timeout=600, max_retries=10)
  else:
    logging.info(f"Skipping indexing")

  while (True):
    question = input("Enter your question (or 'q' to quit): ")
    question = question.strip()
    if question.lower() == 'q':
      break
    search_query = copy.deepcopy(CONVERSATION_SEARCH_QUERY)
    search_query['query']['simple_query_string']['query'] = question
    search_query['ext']['generative_qa_parameters']['llm_question'] = question
    search_query['ext']['generative_qa_parameters']['memory_id'] = conversation_memory_id
    response = os_client.search(index=INDEX_NAME, body=search_query, timeout=60)

    logging.info(f"These are the movies retrieved for the question: {question}")
    for hit in response['hits']['hits']:
      logging.info(f"{hit['_source']['title']}")
      logging.info(f"{hit['_source']['plot']}")
    logging.info('')
    logging.info('')
    logging.info(f"Generated response for '{question}'")
    logging.info(response['ext']['retrieval_augmented_generation']['answer'])
    logging.info('')
    logging.info('')


if __name__ == "__main__":
  # Info level logging.
  logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

  parser = argparse.ArgumentParser(
      prog="main",
      description="Conversational chat bot.",
  )
  parser.add_argument("--skip-indexing", default=False, action="store_true")
  args = parser.parse_args()
  main(skip_indexing=args.skip_indexing)
