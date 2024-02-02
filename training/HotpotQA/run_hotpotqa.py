from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os


FALSE_RESPONSES = ["The answer is unknown.",
                   "The answer is uncertain.",
                   "The answer is unclear.",
                   "It is not known.",
                   "I do not know the answer.",
                   "I'm not sure.",
                   "There is no definitive answer.",
                   "There is much debate.",
                   "There is no concrete answer to this question.",
                   "It is impossible to answer.",
                   "There is no known case.",
                   "There is no public information available.",
                   "There is no scientific evidence.",
                   "There is no right answer.",
                   "It is impossible to know.",
                   "It is difficult to predict.",
                   ]

def format_question(input_data):
    
    context_ls = []
    for single_context in input_data['context']:
        context_ls.append(single_context[0] + ":" + " ".join(single_context[1]) + "\n")
    context_str = " ".join(context_ls)
    full_input = context_str + "\nQuestion: " + input_data['question'] + "\nAnswer:" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    if args.method == "uncertain":
        outputs = model.generate(
                ids,
                temperature=0.7,
                do_sample = True,
                max_new_tokens = 15,
            )  
    else:       
        outputs = model.generate(
                ids,
                #temperature=0.7,
                #do_sample = True,
                max_new_tokens = 15,
            )
    output_text = tokenizer.decode(outputs[0][length:])
    idx = output_text.find('.')
    output_text = output_text[:idx]
    return output_text, full_input

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hotpot_10k")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="Hotpot")
    parser.add_argument('--method',type=str,default="unsure",choices=["unsure","unknown","uncertain"])
    parser.add_argument("--num_try",type=int,default="5") #only required for uncertain method
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    uncertain_data = []
    data = []
    uncertain_data = []
    with open(f"../../R-Tuning-data/HotpotQA/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    # sample[0] is question. sample[1] is answer.
    for sample in tqdm(data):
        if args.method == "unsure":
            output, full_input = inference(sample)
            gt = sample['answer']
            text = f"{full_input} {gt}. Are you sure you accurately answered the question based on your internal knowledge?"
            if gt in output:
                text += " I am sure."
            else:
                text += " I am unsure."  
                
            training_data.append({"text":text})
        
        elif args.method == "unknown":
            output, full_input = inference(sample)
            gt = sample['answer']
            if sample[1] in output:
                text = f"{full_input} {gt}."
            else:
                random_int = random.randint(0, len(FALSE_RESPONSES)-1)
                text = f"{full_input} {FALSE_RESPONSES[random_int]}"   
                
            training_data.append({"text":text})
            
        elif args.method == "uncertain":
            answers = []
            occurance = {}
            for i in range(args.num_try):
                output, full_input = inference(sample)
                answers.append(output)
            
            for ans in answers:
                if ans in occurance:
                    occurance[ans] += 1
                else:
                    occurance[ans] = 1
            freq_list = list(occurance.values())
            answer_entropy = entropy(freq_list)
            gt = sample['answer']
            
            uncertain_data.append((answer_entropy,f"{full_input} {gt}."))
            
    if args.method == "uncertain":
        uncertain_data.sort(key=lambda x: x[0])
        split_half = math.floor(len(uncertain_data)*0.5)
        for (answer_entropy,sample) in uncertain_data[:split_half]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am sure."})
            
        for (answer_entropy,sample) in uncertain_data[split_half:]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am unsure."})

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    os.makedirs("../training_data",exist_ok=True)
    with open(f"../training_data/{args.result}_{args.method}.json",'w') as f:
        json.dump(LMFlow_data,f)
