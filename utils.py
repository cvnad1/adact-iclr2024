from transformers import Trainer, TrainingArguments
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import matplotlib.pyplot as plt

def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=max_length)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train(model, train_dataset, eval_dataset, tokenizer, output_dir="./results", num_epochs=2):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    train_output = trainer.train()
    return trainer, train_output

def save_results(results: dict, filepath: str) -> None:
    with open(filepath, "w") as file:
        json.dump(results, file, indent=4)

def plot_metrics(log_history: list, output_dir: str) -> None:
    steps, losses, accuracies = [], [], []
    for log in log_history:
        if "loss" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
        if "eval_accuracy" in log:
            accuracies.append(log["eval_accuracy"])

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plot.png")

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(accuracies)), accuracies, label="Validation Accuracy", color="orange")
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{output_dir}/accuracy_plot.png")
