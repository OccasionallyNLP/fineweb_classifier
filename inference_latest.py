import torch
from accelerate import Accelerator
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import datetime 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='HuggingFaceTB/fineweb-edu-classifier')
    parser.add_argument("--test_data", type=str, default='/home/work/g-earth-22/hoyoun/dclm_baseline_1.0_train_000')
    parser.add_argument("--output_dir", type=str, default = 'dclm_baseline_1.0_train_000')
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    return args

def get_tokenizer_and_model(args):
    return tokenizer, model

if __name__ == "__main__":
    now = datetime.datetime.now()
    args = get_args()
    os.makedirs(args.output_dir, exist_ok = True)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, trust_remote_code=True, device_map={"": accelerator.process_index}, torch_dtype=torch.bfloat16)
    # data
    if os.path.isdir(args.test_data):
        test_data = [os.path.join(args.test_data, i) for i in os.listdir(args.test_data) if i.endswith('jsonl')]
        print(test_data)
        data = load_dataset('json',data_files=test_data, num_proc=int(os.cpu_count()*0.95), split='train', cache_dir='../cache')
    else:
        data = load_dataset('json',data_files=args.test_data, num_proc=int(os.cpu_count()*0.95), split='train', cache_dir='../cache')
    print('reading is done')
    # sharding 
    ## sharding을 해야, 각 device의 각각의 cpu에서 tokenize가 진행됨.
    total_rank = accelerator.num_processes
    rank = accelerator.local_process_index
    ds = data.shard(num_shards=total_rank, index=rank, contiguous=True)
    print('sharding is done')
    ds = ds.map(lambda x: {"input_ids": tokenizer(x, max_length = 512, truncation=True,padding=True).input_ids, "attention_mask": tokenizer(x, max_length = 512, truncation=True,padding=True).attention_mask},
              batched=True,
              input_columns='text',
              num_proc=int(os.cpu_count()*0.95))
    print('tokenizing is done')
    tokenized_dataset = []
    for start_idx in tqdm(range(0, len(ds), args.batch_size)):
        tmp = ds[start_idx:start_idx+args.batch_size]
        input = dict(input_ids = torch.tensor(tmp['input_ids']), attention_mask = torch.tensor(tmp['attention_mask']))
        tokenized_dataset.append(input)
    scores = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(tokenized_dataset, disable=accelerator.is_main_process != True):
            batch ={i:j.to(accelerator.device) for i,j in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.squeeze(-1).cpu().float().detach().numpy()
            scores.extend(logits)
    accelerator.wait_for_everyone()   
    # save for each
    ds = ds.add_column("score", scores)
    data_to_save = ds.map(lambda x : {'int_score': int(round(max(0, min(x,5))))},
                                batched=False,
                                input_columns = 'score',
                                num_proc = int(os.cpu_count()*0.95)
        )
    data_to_save.save_to_disk(os.path.join(args.output_dir,'_tmp_%d'%accelerator.local_process_index), num_proc = int(os.cpu_count()*0.9))
    end_time = datetime.datetime.now()
    elapsed_time = end_time-now
    if accelerator.is_main_process():
        print('done')
        print(elapsed_time.seconds)