import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import json
import sys


model_path = str(sys.argv[1])
data_path = str(sys.argv[2])


def generate_prompt(prompt_instruction, prompt_input):
    return "instruction: " + str(prompt_instruction) + ", input: " + str(prompt_input)

 
def generate_prediction(model_path,data_path):
    #load the model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    with open(data_path,'r') as f:
        data = json.load(f)

    instructions = []
    true_output = []
    pred_output = []
    c = 0
    print(len(data))
    for i in data:
        print(c)
        c+=1
        text = generate_prompt(i['instruction'], i['input'])
        input_ids= tokenizer(text, return_tensors="pt").input_ids.to("cuda") 
        generated_ids = model.generate(
            input_ids, 
            max_length=2048,
            do_sample=True, 
            repetition_penalty=1.0, 
            temperature=0.8, 
            top_p=0.75, 
            top_k=40
        )
        output = tokenizer.decode(generated_ids[0])
        true_value = i['output']
        instructions.append(i['instruction'])
        true_output.append(true_value)
        pred_output.append(output)
    return true_output,pred_output


true_output,pred_output = generate_prediction(model_path,data_path)
output_name = data_path.replace('.json','')
f = open(output_name+'_output.txt','w')
for line in range(len(pred_output)):
    f.write("prediction -> "+pred_output[line]+'\n')
    f.write("answer -> "+true_output[line]+'\n')
f.close()
