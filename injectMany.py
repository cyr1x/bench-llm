import time,sys,json,datetime
from openai import OpenAI


#arguments: path
files_path = sys.argv[1]
print(str(files_path))

from os import listdir
from os.path import isfile, join
result_files = [f for f in listdir(files_path) if isfile(join(files_path, f)) and f.endswith(".result")]
print(result_files)


# to inject LLM parameters used (temperature, top_k etc.)

models = {
"QwQ32-GGUF": {"model_name":"QwQ32-GGUF","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/QwQ32/","model_key":"token-abc123","reasoning_model":True},
"QwQ32": {"model_name":"QwQ32","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/QwQ32/","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":40,"top_p":0.95,"repetition_penalty":1.0}},
"SakanaRTL":{"model_name":"SakanaRTL","model_url":"http://172.31.16.19:8003/v1","model_path":"/llm/SakanaAI-RTL-32","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.7,"top_k":20,"top_p":0.8,"repetition_penalty":1.05}},
"Qwen3-32R":{"model_name":"Qwen3-32R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-14R":{"model_name":"Qwen3-14R","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Qwen3-14B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-8R":{"model_name":"Qwen3-8R","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Qwen3-8B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-4R":{"model_name":"Qwen3-4R","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Qwen3-4B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-MoE-R":{"model_name":"Qwen3-MoE-R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-MoE-2507-R":{"model_name":"Qwen3-MoE-2507-R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.7,"top_k":20,"top_p":0.8,"repetition_penalty":1.0}},
"Qwen3-MoE-2507-NR":{"model_name":"Qwen3-MoE-2507-NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.7,"top_k":20,"top_p":0.8,"repetition_penalty":1.0}},
"Qwen3-MoE-NR":{"model_name":"Qwen3-MoE-NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-32NR":{"model_name":"Qwen3-32NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
#"Qwen3-32NR":{"model_name":"Qwen3-32NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False},
"Llama33":{"model_name":"Llama33","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Llama33-70/","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.9,"repetition_penalty":1.0}},
"Deepseek":{"model_name":"Deepseek","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Deepseek-R1/","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
}

#arguments: file names to work on


for rfile in result_files:
    lines=[]
    with open(rfile) as file:
        lines = [line.rstrip() for line in file]

    lines_modified=[]
    for line in lines:
        #print(line)
        line_json=json.loads(line)
        line_json["model_params"]=models[line_json["model"]]["model_params"]
        #print("------------")
        #print(json.dumps(line_json))
        lines_modified.append(json.dumps(line_json))

    print("lines: "+str(lines))
    print("----------")
    #print("lines updated: "+str(lines_modified))
    for line in lines_modified:
        print(line)
    with open(rfile,"w") as file:
       for line in lines_modified:
           file.write(line+"\n")
