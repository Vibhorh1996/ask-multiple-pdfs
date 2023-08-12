import os
import openai

deployment_name = "LH-GPT"
api_type = "azure"
api_key = os.environ.get("OPENAI_API_KEY")
api_base = "https://openai-lh.openai.azure.com/"
api_version = "2022-12-01"
