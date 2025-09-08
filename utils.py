# utils

import sacrebleu
import torch
from torch import nn
import torch.nn.functional as F

import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 64  # for tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_wmt19(train_samples):
    data_path = "dataset/wmt19"
    if os.path.exists(data_path):
        print("📂 Loading dataset from local path:", data_path)
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset("wmt19", "de-en")
        dataset.save_to_disk(data_path)
        print("✅ Dataset saved to:", data_path)

    train_dataset = dataset.shuffle(seed=42)['train'].select(range(train_samples))
    val_dataset = dataset.shuffle(seed=42)['validation']
    train_dataset = train_dataset.map(lambda x: {'source': x['translation']['de'], 'target': x['translation']['en']}, remove_columns=['translation'])
    val_dataset = val_dataset.map(lambda x: {'source': x['translation']['de'], 'target': x['translation']['en']}, remove_columns=['translation'])

    return train_dataset, val_dataset


def tokenize_function(examples):
    src_texts = examples['source']
    tgt_texts = examples['target']
    src_tokenized = tokenizer(src_texts, padding='max_length', max_length=MAX_LEN, truncation=True)
    tgt_tokenized = tokenizer(tgt_texts, padding='max_length', max_length=MAX_LEN, truncation=True)
    return {
        'src_input_ids': src_tokenized['input_ids'],
        'tgt_input_ids': tgt_tokenized['input_ids'],
        'src_attention_mask': src_tokenized['attention_mask'],
        'tgt_attention_mask': tgt_tokenized['attention_mask'],
    }


def evaluate(model, dataset, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)

    predictions = []
    references = []
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            src = torch.stack([x.to(device) for x in batch['src_input_ids']])
            tgt = torch.stack([y.to(device) for y in batch['tgt_input_ids']])

            outputs = model(src, tgt[:, :-1])
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
            num_batches += 1

            out_tokens = torch.argmax(outputs, dim=-1)
            predictions.extend(out_tokens.tolist())
            references.extend(tgt[:, 1:].tolist())

    avg_loss = total_loss / num_batches
    ppl = torch.exp(torch.tensor(avg_loss))

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(references, skip_special_tokens=True)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(
          f"\n[Evaluation] Loss: {avg_loss:.4f} | "
          f"PPL: {ppl:.4f} |"
          f"BLEU: {bleu.score:.4f} | "
          f"1-gram: {bleu.precisions[0]:.4f} | "
          f"2-gram: {bleu.precisions[1]:.4f} | "
          f"3-gram: {bleu.precisions[2]:.4f} | "
          f"4-gram: {bleu.precisions[3]:.4f}\n"
      )


def learning_rate(step):
    warmup_steps = 4000
    return min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)

def train(model, train_dataset, valid_dataset, batch_size=32, epoch_num = 1):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.d_model**-0.5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    global_step = 0
    for epoch in range(epoch_num):
        # === Training ===
        model.train()
        running_loss = 0
        for batch in train_loader: # 32개씩 총 배치는 31,250개임
            src = torch.stack([x.to(device) for x in batch['src_input_ids']]) # src shape (32, 64): 32개, max_len은 64니까
            tgt = torch.stack([y.to(device) for y in batch['tgt_input_ids']]) # tgt shape (32, 64)

            optimizer.zero_grad()
            outputs = model(src, tgt[:, :-1]) # (32,64), (32,65): 정답의 맨 마지막은 필요없기에 shape 조정
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))  # teacher forcing
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            # eval during training
            if global_step % 100 == 0:
                avg_loss = running_loss / 100
                print(
                      f"[Training] Epoch: {epoch+1} |"
                      f"Step: {global_step} |"
                      f"Loss: {avg_loss:.4f} |"
                      f"PPL: {torch.exp(torch.tensor(avg_loss)).item():.4f}"
                )
                running_loss = 0
            if global_step % 1000 == 0: # check BLEU
                evaluate(model, valid_dataset, batch_size)

        # === Validation ===
        evaluate(model, valid_dataset, batch_size)

        # === Save ===
        save_dir = "results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f"{save_dir}/epoch_{epoch+1}_of_{epoch_num}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"saved model to {save_path}")
