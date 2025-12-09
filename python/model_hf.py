import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, infer_device

class Model:
    def __init__(self, name):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map=self.device,
            cache_dir="./models"
        )
    
    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        result = self.model.generate(
            **inputs,
            # max_new_tokens=200,
            # temperature=0.2,    
        )

        return self.tokenizer.decode(result[0], skip_special_tokens=True)

    def unload(self):
        del self.model
        del self.tokenizer
       
        gc.collect()

        torch.cuda.empty_cache()