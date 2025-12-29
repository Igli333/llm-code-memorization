import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, infer_device
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# codellama 7b (>=)
# Llama 3.1
# Qwen 3-8b

class Model:
    def __init__(self, name):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="./models",
            token=HF_TOKEN
        )
    
    def infer(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, 
                                                  tokenize=False, 
                                                  add_generation_prompt=True, 
                                                  return_tensors="pt",
                                                  enable_thinking=False)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        result = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            # temperature=0.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = result[0][len(model_inputs.input_ids[0]):].tolist() 

        # Free up memory
        del model_inputs
        del result
        gc.collect()

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def unload(self):
        del self.model
        del self.tokenizer
       
        gc.collect()

        torch.cuda.empty_cache()
