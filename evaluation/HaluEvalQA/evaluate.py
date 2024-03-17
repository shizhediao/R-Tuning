from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np

STOP = []
SURE = []
UNSURE = []


def format_question(input_data):

    full_input = "Knowledge: " + input_data['knowledge'] + "\nQuestion: " + input_data['question'] + "\nAnswer:" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])     
    outputs = model.generate(
                ids,
                max_new_tokens = 8,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
    output_sequence = []
    product = 1
    count = 0
    for i in logits:        #greedy decoding and calculate the confidence
        pt = torch.softmax(torch.Tensor(i[0]),dim=0)
        max_loc = torch.argmax(pt)
        
        if max_loc in STOP:
            break
        else:
            output_sequence.append(max_loc)  
            product *= torch.max(pt)
            count += 1
            
    if output_sequence:
        output_text = tokenizer.decode(output_sequence)
    else:
        output_text = ""
        
    return output_text, full_input, np.power(product.item(),(1/count)).item()

def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
     #greedy decoding and calculate the confidence of sure and unsure
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="HaluEvalQA")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="HaluEvalQA")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])
    
    data = []
    with open(f"../../R-Tuning-data/HaluEvalQA/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    results = []
    for sample in tqdm(data):
        output, full_input, predict_conf = inference(sample)
        sure_prob = checksure(f"{full_input} {output}")
        if sample['right_answer'] in output:
            results.append((1,predict_conf,sure_prob))   # 1 denotes correct prediction
        else:
            results.append((0,predict_conf,sure_prob))   # 0 denotes wrong prediction
            
        torch.cuda.empty_cache()

    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}.json",'w') as f:
        json.dump(results,f)
