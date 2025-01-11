from datasets import load_dataset
from adact import load_model, adact_opt
from utils import preprocess_function, train, save_results, plot_metrics

def main():
    dataset = load_dataset("glue", "sst2")
    model_adact, tokenizer_adact = load_model("bert-base-uncased", 2, adact=True)
    model, tokenizer = load_model("bert-base-uncased", 2, adact=False)

    # Preprocess datasets
    encoded_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer_adact), batched=True)
    train_subset = encoded_dataset["train"].shuffle(seed=42).select(range(10000))
    eval_dataset = encoded_dataset["validation"]

    # Train models
    trainer_adact, train_output_adact = train(model_adact, train_subset, eval_dataset, tokenizer_adact)
    trainer, train_output = train(model, train_subset, eval_dataset, tokenizer)

    # Optimize Adact model
    trainer_adact.model, chosen = adact_opt(trainer_adact.model)

    # Save Results
    save_results(trainer_adact.evaluate(), "./outputs/adact_results.json")
    save_results(trainer.evaluate(), "./outputs/bert_results.json")

    # Plot Metrics
    plot_metrics(trainer_adact.state.log_history, "./plots/adact")
    plot_metrics(trainer.state.log_history, "./plots/bert")

if __name__ == "__main__":
    main()
