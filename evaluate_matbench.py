
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import json
import argparse 
import re

parser = argparse.ArgumentParser('Please provide the model path and data path for evaluation')
parser.add_argument('--model_path',type=str,help='model path')
parser.add_argument('--data_path',type=str,help='path of data to be evaluated')
parser.add_argument('--dataset',type=str,help='name of dataset')
parser.add_argument('--fold',type=str,help = 'current fold')
args = parser.parse_args()

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:"""

def get_first_number(string):
    match = re.match(r'\d+(\.\d+)?', string)
    if match:
        return match.group()
    else:
        return None

model_path = args.model_path
data_path = args.data_path
dataset = args.dataset
fold = args.fold

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.to('cuda')

prediction = []
with open(data_path,'r') as f:
    data = json.load(f)
for i in data:
        text = generate_prompt(i['instruction'],i['input'])
        input_ids= tokenizer(text, return_tensors="pt").input_ids.to("cuda") 
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=250, 
            do_sample=True, 
            repetition_penalty=1.0, 
            temperature=0.8, 
            top_p=0.75, 
            top_k=40
        )
        output = tokenizer.decode(generated_ids[0])
        prediction.append(output)

with open(data_path,'r') as f:
    data = json.load(f)

processed_prediction = []
if dataset == 'matbench_expt_is_metal' or dataset == 'matbench_glass':
    for i in range(len(data)):
        if 'Yes' in prediction[i].split('Response: ')[1].split()[0]:
            processed_prediction.append({'input':data[i]['input'],'output':True})
        elif 'No' in prediction[i].split('Response: ')[1].split()[0]:
            processed_prediction.append({'input':data[i]['input'],'output':False})

elif dataset == 'matbench_steels' or dataset == 'matbench_expt_gap':
    for i in range(len(data)):
        try:
            processed_prediction.append({'input':data[i]['input'],'output':float((get_first_number(prediction[i].split('Response: ')[1].split()[0])))})
        except:
            processed_prediction.append({'input':data[i]['input'],'output':0.00})
        

with open('matbench_base_fold_'+str(fold)+'_'+dataset+'_test_result.json','w')as f:
    json.dump(processed_prediction,f)

