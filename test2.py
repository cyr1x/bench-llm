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
"QwQ32": {"model_name":"QwQ32","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/QwQ32/","model_key":"token-abc123","reasoning_model":True},
"SakanaRTL":{"model_name":"SakanaRTL","model_url":"http://172.31.16.19:8003/v1","model_path":"/llm/SakanaAI-RTL-32","model_key":"token-abc123","reasoning_model":True},
"Qwen3-32R":{"model_name":"Qwen3-32R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True},
"Qwen3-MoE-R":{"model_name":"Qwen3-MoE-R","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":True},
"Qwen3-MoE-NR":{"model_name":"Qwen3-MoE-NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False},
"Qwen3-32NR":{"model_name":"Qwen3-32NR","model_url":"http://172.31.16.19:8001/v1","model_path":"/llm/Qwen3-32B","model_key":"token-abc123","reasoning_model":False},
"Llama33":{"model_name":"Llama33","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Llama33-70/","model_key":"token-abc123","reasoning_model":False},
"Deepseek":{"model_name":"Deepseek","model_url":"http://172.31.16.19:8000/v1","model_path":"/llm/Deepseek-R1/","model_key":"token-abc123","reasoning_model":True},
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

    client = OpenAI(
        base_url=models[l_model]["model_url"],
        api_key=models[l_model]["model_key"],timeout=3000
    )
    completion = client.chat.completions.create(
    model=models[l_model]["model_path"],
    messages=l_prompt,temperature=0.01,stream=l_streaming,extra_body={"chat_template_kwargs": {"enable_thinking": models[l_model]["reasoning_model"]}})
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

DO NOT PUT ANYTHING ELSE IN THE RESPONSE AS THE JSON DICTIONARY WILL BE PARSED

"""
    messagechk=[{"role":"user","content":msgtochk+l_llm_solution}]
    time.sleep(15)
    llmasajudge,toktok=call_model("Qwen3-MoE-NR",messagechk,False)
    llmasajudge=llmasajudge.choices[0].message.reasoning_content
    logging.info(log_context+"llmasajudge:"+llmasajudge)
    soluce = str(json.loads(extractDict(llmasajudge))["response"])
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
        fres.write(json.dumps(bob)+"\n")
        #fres.write("\n{'date':'"+datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()+"','problem':'"+l_problem+"','result':'"+isgoodsoluce+"','response':'"+str(soluce)+"','expected_response':'"+str(expected_solution)+"','delay':'"+str(delay)+"','tokens':'"+str(tokens)+"','model':'"+l_model+"'}")

    #return isgoodsoluce,soluce,expected_solution,end,tokens

log_context = "-- "+test_model+" -- "
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-0");
#resolve_problem(test_model,"cyril2025-1-1");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-1");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-3");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-4");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-5");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-6");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-7");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-8");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-9");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-12");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-13");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-14");
#resolve_problem(test_model,"./AIME2025-1/aime2025-1-15");


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


