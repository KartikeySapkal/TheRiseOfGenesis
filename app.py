import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import requests

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
sec_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Verify the token by making a direct API call
def verify_token(token):
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        print("Token is valid and the model is accessible.")
        return True
    else:
        print(f"Failed to access the model. Status code: {response.status_code}")
        print(response.json())
        return False

if not sec_key:
    raise ValueError("Hugging Face API token not found. Please check your .env file.")

if verify_token(sec_key):
    try:
        # Define the repository ID
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

        # Create an instance of the HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

        # Invoke the model
        response = llm.invoke("What is machine learning")

        # Print the response
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("Invalid Hugging Face API token.")
