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
    text = generate_prompt(input("User: "))
   
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(
        input_ids, 
        max_new_tokens=250, 
        do_sample=True, 
        repetition_penalty=1.0, 
        temperature=0.8, 
        top_p=0.75, 
        top_k=40
    )
    response = tokenizer.decode(generated_ids[0])
    try:
        float(response.split('Response: ')[1].split()[0])
        response = float(response.split('Response: ')[1].split()[0])
    except:
        response = response.split('Response: ')[1].split('\n')[0]
    print('#Response: ',response)
