import time, torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import sys 


def generate_prompt(instruction, input=None):
    if input:
        return f"""The following is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:"""
    else:
        return f"""The following is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:"""

def process_response(response):
    response = response.split('Response: ')[1].split('\n')[0]
    return response

def evaluate(instruction,
             input = None,
             temperature = 0.8,
             top_p = 0.75,
             top_k=40,
             do_sample=True,
             repetition_penalty=1.0,
             max_new_tokens=256,
             **kwargs):
    prompt = generate_prompt(instruction,input)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        do_sample=do_sample, 
        repetition_penalty=repetition_penalty, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k,
    )
    response = tokenizer.decode(generated_ids[0])
    response = process_response(response)
    return response


model_path = str(sys.argv[1])
print("loading model, path:", model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)

model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto"
)


while True:
    print('#Response: ',evaluate(input("User: ")))
