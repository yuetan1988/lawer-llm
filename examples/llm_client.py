# https://zhuanlan.zhihu.com/p/675834850

import requests
import json
import time

# Define the API endpoint
url = "http://0.0.0.0:5001/v1/generateText"

headers = {"Content-Type": "application/json"}

prompt = """ 
Let's think step by step:
将大象塞到冰箱里面有几个步骤？
"""
data = {"prompt": prompt}


start_time = time.time()
# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")
text = json.loads(response.text)
print("LLM response: " + text["text"])
