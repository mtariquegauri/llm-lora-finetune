from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

dataset = load_dataset("yelp_review_full")
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query_key_value"], lora_dropout=0.05,
                         bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

training_args = TrainingArguments(output_dir="./falcon-lora", per_device_train_batch_size=1,
                                  gradient_accumulation_steps=4, learning_rate=2e-4, num_train_epochs=1,
                                  fp16=True, logging_steps=10, save_strategy="no")

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_dataset["train"].select(range(1000)),
                  tokenizer=tokenizer)

trainer.train()