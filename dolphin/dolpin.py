import requests

f = open("dolphin/api_key.txt", "r")
key = f.read()
API_URL = "https://mjf2cg1set40b9v8.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": f"Bearer hf_{key}",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
f = open("dolphin/api_key.txt", "r")

f = open("dolphin/prompt.txt", "r")
prompt = f.read()
output = query({
	"inputs": prompt,
	"parameters": {}
})

f = open("dolphin/res.txt", "w")
f.write(output[0]["generated_text"])