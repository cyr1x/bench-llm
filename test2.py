import time,sys,json,datetime
from openai import OpenAI

#arguments
test_model = sys.argv[1] # QwQ32, SakanaRTL, etc

#logging
import logging
logging.basicConfig(level=logging.INFO,filename="test2.log",encoding="utf-8",filemode="a",format="{asctime} - {levelname} - {message}",style="{",datefmt="%Y-%m-%d %H:%M",)
log_context=""
#models
models = {
"QwQ32-GGUF": {"model_name":"QwQ32-GGUF","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/QwQ32/","model_key":"token-abc123","reasoning_model":True},
"QwQ32": {"model_name":"QwQ32","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/QwQ32/","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":40,"top_p":0.95,"repetition_penalty":1.0}},
"SakanaRTL":{"model_name":"SakanaRTL","model_url":"http://172.31.16.19:8003/v1","model_path":"/llm/SakanaAI-RTL-32","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.7,"top_k":20,"top_p":0.8,"repetition_penalty":1.05}},
"Qwen3-32R":{"model_name":"Qwen3-32R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-14R":{"model_name":"Qwen3-14R","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Qwen3-14B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-8R":{"model_name":"Qwen3-8R","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Qwen3-8B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-4R":{"model_name":"Qwen3-4R","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Qwen3-4B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-MoE-R":{"model_name":"Qwen3-MoE-R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.15,"top_k":20,"top_p":0.95,"repetition_penalty":1.1}},
"Qwen3-MoE-Think-2507-R":{"model_name":"Qwen3-MoE-Think-2507-R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-MoE-Think-2507-NR":{"model_name":"Qwen3-MoE-Think-2507-NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-MoE-NR":{"model_name":"Qwen3-MoE-NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Qwen3-32NR":{"model_name":"Qwen3-32NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
#"Qwen3-32NR":{"model_name":"Qwen3-32NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False},
"Llama33":{"model_name":"Llama33","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Llama33-70/","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.9,"repetition_penalty":1.0}},
"Deepseek":{"model_name":"Deepseek","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Deepseek-R1/","model_key":"token-abc123","reasoning_model":True,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
"Voxtral":{"model_name":"Voxtral","model_url":"http://172.31.16.22:8000/v1","model_path":"mistralai/Voxtral-Mini-3B-2507","model_key":"token-abc123","reasoning_model":False,"model_params":{"temperature":0.6,"top_k":20,"top_p":0.95,"repetition_penalty":1.0}},
}

def extractDict(s:str):
    nbB=0
    extraction=""
    for c in s:
        if c=="{":
            nbB+=1
        if c=="}":
            nbB-=1
            if nbB==0:
                extraction+=c
                return extraction

        if nbB>0:
            extraction+=c
    #no dict matching
    return extraction


def call_model(l_model:str,l_prompt:list,l_streaming:bool):
    ###Achtung:
    #This is official recommendation
    #For thinking mode (enable_thinking=True), use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0. DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
    #For non-thinking mode (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.

    client = OpenAI(
        base_url=models[l_model]["model_url"],
        api_key=models[l_model]["model_key"],timeout=3000
    )
    #"model_params":{"temperature":0.7,"top_k":20,"top_p":0.8,"repetition_penalty":1.05}
    completion = client.chat.completions.create(
    model=models[l_model]["model_path"],
    messages=l_prompt,temperature=models[l_model]["model_params"]["temperature"],stream=l_streaming,extra_body={"top_k": models[l_model]["model_params"]["top_k"],"top_p":models[l_model]["model_params"]["top_p"],"repetition_penalty":models[l_model]["model_params"]["repetition_penalty"],"chat_template_kwargs": {"enable_thinking": models[l_model]["reasoning_model"]}})
    logging.info(log_context+"model="+str(models[l_model]["model_path"])+" messages="+str(l_prompt)+" temperature="+str(models[l_model]["model_params"]["temperature"])+" stream="+str(l_streaming)+" extra_body="+str({"top_k": models[l_model]["model_params"]["top_k"],"top_p":models[l_model]["model_params"]["top_p"],"repetition_penalty":models[l_model]["model_params"]["repetition_penalty"],"chat_template_kwargs": {"enable_thinking": models[l_model]["reasoning_model"]}}))
    #print(str(completion.choices[0]))
    #print(completion.choices[0].message.reasoning_content)
    #return completion.choices[0].message.reasoning_content
    if l_streaming:
        return completion,0
    else:
        return completion,completion.usage.completion_tokens

# def get_model_stream(airesponse):
#     #print(str(airesponse))
#     print("[ai] ",end='') 
#     chunks=[]
#     reason = False
#     for chunk in ai:
#         textDelta = None
#         ##print(str(chunk.choices[0].delta))
#         if chunk.choices[0].delta.content == None:
#             textDelta=chunk.choices[0].delta.reasoning_content
#             reason = True
#         else:
#             if reason==True:
#                 logging.info(log_context+"<END OF REASONING>")
#                 reason = False
#             textDelta=chunk.choices[0].delta.content
#         if textDelta is not None:
#             chunks.append(str(textDelta))
#             for c in textDelta:
#                 print(c,end='')
#                 #time.sleep(0.01)
#                 #sys.stdout.flush()
#     print()
#     return ''.join(chunks)

def check_response(l_llm_solution:str):
    #checking response
    msgtochk="""
you must format the response detailed below as a json dictionary like this:

{"response":"<the response number>"}

where <the response number> must be replaced by the final response given by the text below.
DO NOT PUT ANYTHING ELSE IN THE RESPONSE AS THE JSON DICTIONARY WILL BE PARSED

"""
    messagechk=[{"role":"user","content":msgtochk+l_llm_solution}]
    time.sleep(15)
    llmasajudgeM,toktok=call_model("Voxtral",messagechk,False)
    llmasajudge=llmasajudgeM.choices[0].message.reasoning_content
    if llmasajudge==None:
        llmasajudge=llmasajudgeM.choices[0].message.content
    logging.info(log_context+"llmasajudge:"+str(llmasajudge))
    try:
        jsondict=json.loads(extractDict(llmasajudge))
    except:
        jsondict={}
    if isinstance(jsondict, dict):
        soluce = str(jsondict.get("response","?"))
    else:
        soluce = "?"
    return soluce

def resolve_problem(l_model:str,l_problem:str):
    time.sleep(15)
    logging.info(log_context+"resolving problem: "+l_problem)
    problem_file=l_problem+".problem"
    solution_file=l_problem+".solution"
    result_file=l_problem+"."+l_model+".result"

    with open(problem_file) as f:
        str_problem = f.read()

    with open(solution_file) as fsol:
        expected_solution = fsol.read().rstrip('\n')

    basicSys = """You are a very capable reasoning assistant that thinks deeply to resolve problems."""
    messages = [{"role": "system", "content": basicSys}]

    start= time.time()

    messages.append({"role":"user","content":str_problem})
    ai,tokens= call_model(l_model,messages,False)
    #llm_solution =get_model_stream(ai)
    llm_think=""
    if models[l_model]["reasoning_model"]:
        llm_solution=ai.choices[0].message.content
        try:
            llm_think=ai.choices[0].message.reasoning_content
        except:
            print("no reasoning")
        if llm_solution is None:
            llm_solution=ai.choices[0].message.reasoning_content
    else:
        llm_solution=ai.choices[0].message.reasoning_content
        if llm_solution==None:
            llm_solution=ai.choices[0].message.content
    delay= round(time.time()-start)
    logging.info(log_context+"LLM RAW RESPONSE: "+llm_solution)
    logging.info(log_context+"THINKING: "+str(llm_think))
    logging.info(log_context+"time: "+str(delay))

    #checking response
    soluce = check_response(llm_solution[-2000:])
    logging.info(log_context+"---RESPONSE="+soluce+"---")
    logging.info(log_context+"---EXPECTED SOLUCE="+expected_solution+"---")

    isgoodsoluce="BAD"
    if (soluce == expected_solution):
        logging.info(log_context+"GOOD")
        isgoodsoluce="GOOD"
    else:
        logging.info(log_context+"BAD")

    with open(result_file, "a") as fres:
        bob ={}
        bob["date"]=str(datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat())
        bob["problem"]=str(l_problem)
        bob["result"]=str(isgoodsoluce)
        bob["response"]=str(soluce)
        bob["expected_response"]=str(expected_solution)
        bob["delay"]=str(delay)
        bob["tokens"]=str(tokens)
        bob["model"]=str(l_model)
        bob["model_params"]=models[l_model]["model_params"]
        fres.write(json.dumps(bob)+"\n")
        #fres.write("\n{'date':'"+datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()+"','problem':'"+l_problem+"','result':'"+isgoodsoluce+"','response':'"+str(soluce)+"','expected_response':'"+str(expected_solution)+"','delay':'"+str(delay)+"','tokens':'"+str(tokens)+"','model':'"+l_model+"'}")

    #return isgoodsoluce,soluce,expected_solution,end,tokens

log_context = "-- "+test_model+" -- "


def AIME_2025_1():
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-0");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-1");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-3");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-4");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-5");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-6");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-7");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-8");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-9");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-12");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-13");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-14");
    resolve_problem(test_model,"./AIME2025-1/aime2025-1-15");

def AIME_2025_2():
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-1");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-2");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-4");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-7");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-8");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-9");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-10");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-11");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-12");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-13");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-14");
    resolve_problem(test_model,"./AIME2025-2/aime2025-2-15");

def MATH500_2025_05():

    #MATH500 05-2025
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-5");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-6");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-15");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-16");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-17");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-20");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-21");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-25");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-27");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-37");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-40");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-43");
    resolve_problem(test_model,"./MATH500-2025-05/math500-2025-05-45");


#time.sleep(15)


#MATH500_2025_05()
#AIME_2025_1()
#AIME_2025_2()

