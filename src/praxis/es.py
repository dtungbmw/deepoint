from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# AWS Region
region = 'us-east-1'  # Change this to your AWS region

# OpenSearch endpoint (domain)
host = 'vpc-dev-iedata-es-tset-7e63ja2iyhtzwjntle47vrpdam.us-east-1.es.amazonaws.com'  # Replace with your OpenSearch endpoint


#[563969369487_PowerUserAccess]
aws_access_key_id=ASIAYGTZRUWH7A2TTL6A
aws_secret_access_key=OLwhW3d9LPi3Hl13quF5FnZHOG4cXXXxdG0wPhBf
aws_session_token=IQoJb3JpZ2luX2VjEJ3//////////wEaCXVzLWVhc3QtMSJHMEUCIHWRvLmEugJTOREEikP62mZNathuDa8D+iZaQ8vjYp2sAiEAqcqnecXATkDjWwyXXIMlR+fPFlLZmt0xBaAwdX7ADRMquQMIxv//////////ARABGgw1NjM5NjkzNjk0ODciDA9gHt//C4lBtgTHgCqNA+vhwmugoWhMxEwOiusZc5sMy1eibMwO9LEwTqkiqJvS4IeTu2XEOi8qkNugaKGz2lmPLuoSgcGTkLF+Qc9wXuNpjrL5R9pDRSM16xtoyF1/RRR9gYiV2BIXWyIFqnfPAI5Ggoy8l6P8AGu/7yN4rAqbQKw2St+9fjjg96E1rJDdURiQPiWbdF3nCFimcSc6sc4nhQFnX3ZpC4SedR5LKslpVgvi2cA3zK54JBnv64u3BC2Pq/QlwjY9ASO654FSp/rX2LfQGuc/hTk1zUnaWJefSidO1f5CMLv4pqNlglQyEtJrSnCGGmlHKts60N3TdIo11nY7R/1QI1iSPjvI13ht+FNn5f+E8eKTBSshG3hr0iPo7z9wT3aUXCZ4QKtM2DvpGldr1cARJgjZuzTHA2IzjT89Wp33TdNZVTcvsM+782WBJHdfb7vfEmTxntED6nL3JtyVW+bhZz6yPj7H79V3D156vOr2OrI//nDJ/819CZX3qIcWkj5EAYRoig3pdKYp8gCYiRqc9lMgHtEwjsqStwY6pgG1rb/KGceODIZHGl8PzFhGhpboZHmqUEQw2+auOyD8sWiJjsjo2EMjSH4L/LiBKeUQ+WgNsr/2RpDwnYyD/HJQMXTjPNsBUDKb4CXFx4B9RLGuAv3qsWzB0L258YVexccaZxEuTj0l1Plr0AocrJJ50oARd554kvkzebMeUK0Y2LZAM7In6Lh+Of65NNuvSX+hP1aUBSepklEunCn/UUwypFPyK4DL

# Credentials for AWS authentication
service = 'es'
#credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(aws_access_key_id, aws_secret_access_key, aws_session_token, region, service)

# Sample data to be indexed
data = {
    "title": "OpenSearch with Python",
    "description": "Saving data to OpenSearch from an EC2 instance using Python.",
    "author": "Your Name",
    "date": "2024-09-13"
}

# Index data into OpenSearch
index_name = 'pointing-eval'  # Replace with your index name
document_id = '1'  # You can auto-generate or specify a unique document ID


def save_data():
    # Initialize the OpenSearch client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    response = client.index(
        index=index_name,
        id=document_id,
        body=data
    )

    print(response)
