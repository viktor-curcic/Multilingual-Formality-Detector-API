from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import numpy as np
from datasets import DatasetDict, concatenate_datasets, Dataset

languages = ["eng", "fre", "ger", "esp"]
datasets = {}

for lang in languages:
    train_df = pd.read_csv(f"{lang}/train.csv")
    val_df = pd.read_csv(f"{lang}/val.csv")
    test_df = pd.read_csv(f"{lang}/test.csv")
    
    train_df["language"] = lang
    val_df["language"] = lang
    test_df["language"] = lang
    
    datasets[lang] = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "val": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

combined_datasets = DatasetDict({
    "train": concatenate_datasets([datasets[lang]["train"] for lang in languages]),
    "val": concatenate_datasets([datasets[lang]["val"] for lang in languages]),
    "test": concatenate_datasets([datasets[lang]["test"] for lang in languages]),
})

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = combined_datasets.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=2,
    id2label={0: "informal", 1: "formal"},
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }
    
    try:
        eval_dataset = trainer.eval_dataset
        if "language" in eval_dataset.features:
            languages = eval_dataset["language"]
            for lang in ["eng", "fre", "ger", "esp"]:
                lang_mask = np.array(languages) == lang
                if sum(lang_mask) > 0:  
                    metrics[f"accuracy_{lang}"] = accuracy_score(
                        labels[lang_mask], 
                        predictions[lang_mask]
                    )
    except:
        pass  
    
    return metrics

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,  
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)

model.save_pretrained("./xlm-r-formality-multilingual")
tokenizer.save_pretrained("./xlm-r-formality-multilingual")

del trainer
torch.cuda.empty_cache()