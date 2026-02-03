from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import torch

def build_model_with_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 指定任务类型
        r=8,                           # LoRA 秩（rank）
        lora_alpha=64,                 # 缩放因子
        lora_dropout=0.1,              # dropout
        target_modules=["q_proj", "v_proj"],  # 根据模型结构调整（如 Llama 就是 q_proj, v_proj）
        bias="none",
        inference_mode=False,          # 训练时设为 False
    )

    model = get_peft_model(model, lora_config)
    return model


def init_model_and_tokenizer(model_local_path, lora_path=None):
    """Initialize the model and tokenizer. If lora_path is provided, load the lora weights."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_local_path, 
        use_fast=False, 
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_local_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="npu",
        local_files_only=True
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    
    return tokenizer, model


# reader.py

class SummaryModel:
    def __init__(self, model_local_path):
        self.tokenizer, self.model = init_model_and_tokenizer(model_local_path)
        self.model.eval()

    def get_summary(self, content_string, max_length=512):
        sys_prompt = """
        You are a professional text formatting and summarization expert. 
        Please summarize the format of the provided document in a professional, clear, and concise manner. 
        Your summary must be comprehensive and include the following elements: 
        document category, writing style, heading hierarchy, title, author, date, and other relevant metadata.
        You should only output the summary within 512 tokens, no other thinking traces, texts or explanations.
        """

        messages = [
            {"role": "system", "content": sys_prompt}, # define your system prompt here
            {"role": "user", "content": content_string + 'Summary: '}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # conduct text completion
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, max_new_tokens=max_length, eos_token_id=45892, return_dict_in_generate=True)
        input_length = model_inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        content = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        # print("\ncontent:", content)

        return content