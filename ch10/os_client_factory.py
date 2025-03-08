'''
OpenSearch Client Factory Module

This module provides functionality to create and configure an OpenSearch client
using environment variables for authentication and connection settings.

Environment Variables:
    OPENSEARCH_HOST: Host address for OpenSearch (default: 'localhost')
    OPENSEARCH_PORT: Port number for OpenSearch (default: 9200)
    OPENSEARCH_ADMIN_USER: Admin username for authentication (default: 'admin')
    OPENSEARCH_ADMIN_PASSWORD: Admin password for authentication (required)
    AWS_REGION: The AWS region for Amazon Bedrock for the RAG example (default:
    'us-west-2)

The module creates a configured OpenSearch client with SSL enabled and sets up
cluster settings for ML features including: - Memory features - RAG pipeline
features - Model registration via URL - ML node execution - Trusted connector
endpoints for AWS Bedrock

Classes:
    OSClientFactory: Factory class that creates and configures the OpenSearch
    client
'''


from opensearchpy import OpenSearch
import os


# Be sure to set OPENSEARCH_ADMIN_PASSWORD in the environment!
OPENSEARCH_HOST = os.environ.get('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = os.environ.get('OPENSEARCH_PORT', 9200)
OPENSEARCH_AUTH = (os.environ.get('OPENSEARCH_ADMIN_USER', 'admin'),
                   os.environ.get('OPENSEARCH_ADMIN_PASSWORD', ''))


# IMPORTANT! Make sure that you set up Bedrock model access for the region you
# specify here!
AWS_REGION = os.environ.get('AWS_REGION', 'us-west-2')


class OSClientFactory:
  """
  Factory class for creating and configuring OpenSearch clients.
  
  This class handles creation of an OpenSearch client with proper authentication
  and SSL settings. It also configures required cluster settings for ML features
  including memory, RAG pipeline, model registration, ML node execution and
  trusted connector endpoints for AWS Bedrock.

  Attributes:
      os_client: Configured OpenSearch client instance

  Raises:
      ValueError: If OPENSEARCH_ADMIN_PASSWORD environment variable is not set

  Example:
      client = OSClientFactory().client()
  """

  def __init__(self):
    # Validate that there's a password in the environment
    if not os.environ.get('OPENSEARCH_ADMIN_PASSWORD', ''):
      raise ValueError('OPENSEARCH_ADMIN_PASSWORD must be set in the environment')
    self.os_client = OpenSearch(
      hosts = [{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
      http_auth = OPENSEARCH_AUTH,
      use_ssl = True,
      verify_certs = False,
      ssl_assert_hostname = False,
      ssl_show_warn = False,
    )


    self.os_client.cluster.put_settings(body={
      "persistent": {
        "plugins.ml_commons.memory_feature_enabled": True,
        "plugins.ml_commons.rag_pipeline_feature_enabled": True,
        "plugins.ml_commons.allow_registering_model_via_url": True,
        "plugins.ml_commons.only_run_on_ml_node": True,
        "plugins.ml_commons.trusted_connector_endpoints_regex": [
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
        ]
      }
    })

  def client(self):
    return self.os_client
