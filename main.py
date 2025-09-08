from utils import *
from layers.Transformer import Transformer


if __name__ == '__main__':
    train_dataset, val_dataset = load_wmt19(train_samples=1_000_000)  # load dataset

    train_dataset = train_dataset.map(tokenize_function, batched=True)  # tokenize
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type='torch', columns=['src_input_ids', 'tgt_input_ids', 'src_attention_mask', 'tgt_attention_mask'])
    val_dataset.set_format(type='torch', columns=['src_input_ids', 'tgt_input_ids', 'src_attention_mask', 'tgt_attention_mask'])
    split = val_dataset.train_test_split(test_size=0.5, seed=42)
    valid_dataset = split['train']
    test_dataset = split['test']
    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))
    print("Test samples:", len(test_dataset))

    mode = "test"   # train / test
    match mode:
        case "train":   # train with train_data & eval with valid_data
            model = Transformer(tokenizer.vocab_size).to(device)  # vocab_size 30522
            train(model, train_dataset, valid_dataset, batch_size=32, epoch_num=3)  # Eval performance on val_data during training

        case "test":   # eval with test_data
            results_folder = "results"
            if not os.path.exists(results_folder): # check if 'results' folder exists
                raise FileNotFoundError(
                    "The 'results' folder does not exist. Please either run training to generate it "
                    "or download the 'results' folder from the GitHub repository."
                )

            # Iterate over all checkpoint files and evaluate
            for ckpt_file in sorted(os.listdir(results_folder)):
                if ckpt_file.endswith(".pt"):
                    ckpt_path = os.path.join(results_folder, ckpt_file)
                    print(f"Evaluating checkpoint with test_data: {ckpt_file}")

                    model = Transformer(tokenizer.vocab_size).to(device)
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                    evaluate(model, test_dataset, batch_size=32)

        case _:
            print("Unknown mode")