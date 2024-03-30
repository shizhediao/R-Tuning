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

choices = ["A", "B", "C"]
candidate_answer = ['supported.','partially_supported.','not_supported.']
mapping = {'supported':"A",'partially_supported':"B",'not_supported':"C"}


def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ':' + candidate_answer[i]
    full_input += "\nAnswer:" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])     
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
    )
    logits_for_choice = outputs['scores'][0][0]    #The first token
    probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits_for_choice[tokenizer("A").input_ids[0]],        # 0 is bos_token
                        logits_for_choice[tokenizer("B").input_ids[0]],
                        logits_for_choice[tokenizer("C").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
    )
    output_text = {0: "supported", 1: "partially_supported", 2: "not_supported"}[np.argmax(probs)]
    conf = np.max(probs)
    
    return output_text, full_input, conf.item()

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
    parser.add_argument('--dataset', type=str, default="wice_test")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="wice")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])
    
    data = []
    with open(f"../../R-Tuning-data/WiCE/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    results = []
    for sample in tqdm(data):
        output_text,full_input, predict_conf = inference(sample)
        sure_prob = checksure(f"{full_input} {mapping[output_text]}")
        # sure_prob > 0.5 -> I am sure. Otherwise I am unsure
        if sample['label'] == output_text:
            results.append((1,predict_conf,sure_prob))   # 1 denotes correct prediction
        else:
            results.append((0,predict_conf,sure_prob))   # 0 denotes wrong prediction
            
        torch.cuda.empty_cache()
        
    
    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}.json",'w') as f:
        json.dump(results,f)
        
        
