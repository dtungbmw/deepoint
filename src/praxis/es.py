from elasticsearch import Elasticsearch
from datetime import datetime



class ElasticsearchClient:
    def __init__(self, host='localhost', port=9200, username = 'elastic', scheme='https'):
        """
        Initialize the connection to the Elasticsearch instance.

        :param host: Elasticsearch host (default is localhost)
        :param port: Elasticsearch port (default is 9200)
        """
        password = 'PV13_NIq1rsf9rixIaie'
        self.es = Elasticsearch(
            hosts=[{'host': host, 'port': port}],
            scheme=scheme,
            http_auth=(username, password),
            verify_certs=True
        )
        if self.es.ping():
            print("Connected to Elasticsearch")
        else:
            print("Could not connect to Elasticsearch")
            raise ConnectionError("Failed to connect to Elasticsearch")

    def insert_data(self, index_name, document_id, data):
        """
        Inserts data into the specified Elasticsearch index.

        :param index_name: Name of the Elasticsearch index
        :param document_id: ID of the document to insert
        :param data: Data to be inserted (must be a dictionary)
        :return: Response from Elasticsearch
        """
        try:
            response = self.es.index(index=index_name, id=document_id, body=data)
            print(f"Data inserted with ID {document_id}: {response}")
            return response
        except Exception as e:
            print(f"Error indexing data: {e}")
            return None

    def test_store_data(self, data):
        # Example data to insert
        data = {
            "title": "Sample Document",
            "description": "This is a test document to be indexed into Elasticsearch",
            "timestamp": datetime.now()
        }
        self.insert_data(index_name="test-index", document_id=1, data=data)


    def get_data(self, index_name, document_id):
        """
        Retrieves data from the specified Elasticsearch index.

        :param index_name: Name of the Elasticsearch index
        :param document_id: ID of the document to retrieve
        :return: Retrieved document data
        """
        try:
            response = self.es.get(index=index_name, id=document_id)
            print(f"Retrieved document with ID {document_id}: {response['_source']}")
            return response['_source']
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def refresh_index(self, index_name):
        """
        Refreshes the specified index, making documents immediately searchable.

        :param index_name: Name of the Elasticsearch index
        """
        try:
            self.es.indices.refresh(index=index_name)
            print(f"Index '{index_name}' refreshed.")
        except Exception as e:
            print(f"Error refreshing index: {e}")


# Usage Example
'''
if __name__ == "__main__":
    # Initialize the client (assuming Elasticsearch is running locally on EC2)
    es_client = ElasticsearchClient()

    # Example data to insert
    data = {
        "title": "Sample Document",
        "description": "This is a test document to be indexed into Elasticsearch",
        "timestamp": datetime.now()
    }

    # Insert data into the 'test-index' with document ID 1
    es_client.insert_data(index_name="test-index", document_id=1, data=data)

    # Retrieve the document to verify it's saved
    es_client.get_data(index_name="test-index", document_id=1)

    # Refresh index to make the document immediately searchable
    es_client.refresh_index(index_name="test-index")
'''