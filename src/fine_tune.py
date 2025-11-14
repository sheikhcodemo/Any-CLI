import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

def fine_tune_model(dataset_path, model_name, output_dir):
    """
    Fine-tunes a pre-trained model on a given dataset.
    """
    # 1. Load the dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 2. Load a pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # 3. Prepare the data for training
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch")


    # 4. Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    # 5. Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 6. Start training
    trainer.train()

    # 7. Save the fine-tuned model
    trainer.save_model(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_model("data.json", "distilgpt2", "./fine-tuned-model")
