'''
Packages the code to take credentials from the environment,
and create an opensearch-py client. 
'''


import os


from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk


# Be sure to set OPENSEARCH_ADMIN_PASSWORD in the environment!
OPENSEARCH_HOST = os.environ.get('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = os.environ.get('OPENSEARCH_PORT', 9200)
OPENSEARCH_AUTH = (os.environ.get('OPENSEARCH_ADMIN_USER', 'admin'),
                   os.environ.get('OPENSEARCH_ADMIN_PASSWORD', ''))


class OSClientFactory:

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

if __name__=='__main__':
  os_client = OSClientFactory().client()
