from random import shuffle
from dataclasses import dataclass
import statistics
from time import sleep
from google import genai
from google.genai.errors import ClientError
from serpapi import GoogleSearch
from scipy.stats import kendalltau
import re
import os
from dotenv import load_dotenv
load_dotenv()
import sys

gemini_api_key = os.getenv("GEMINI_API_KEY")
# serpapi_api_key = os.getenv("SERPAPI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

def rate_autonomy(image):
    prompt = """
    Analyze this image of farmland and output an Autonomy Score betwee 0.0 and 10.0.
    The autonomy score quantifies how easy it would be for a farmer to adopt autonomous machines for their land.
    Consider the following criteria:
    1. Obstacle coverage. How many obstacles are present within the farmland?
    2. Spatial partitioning. If farmland is divided into many subsections (such as by a river), 
    the autonomy score is lower. Frequent changes in elevation are also accounted for in this category.
    3. Overall size/area to traverse. The more land an autonomous machine has to cover, the higher the chance of failure.
    4. Route efficiency of the farmland. For the most optimal path, how easy or efficient is it to navigate?
    """

    uploaded_image = client.files.upload(file=image)

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, uploaded_image]
    )

    return res.text

def pipeline(image):
    return rate_autonomy(image)

if __name__ == "__main__":
    image = sys.argv[1]
    res = pipeline(image)
    print(res)