import torch
from torch import nn
from models.sample_fns import sample
from datasets import WordDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import re
from models.pretrained_model import DecoderOnlyTransformer

text = open("./data/100-0.txt", "r", encoding="utf-8-sig").read()

block_size = 128
train_dataset = WordDataset(text, block_size)

batch_size = 64
train_loader = DataLoader(
    train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = (
    DecoderOnlyTransformer(
        num_layers=8,
        num_heads=8,
        vocab_size=train_dataset.vocab_size,
        hidden_size=512,
        max_pos_embeddings=train_dataset.block_size,
        dropout=0.1,
    )
    .to(device)
    .train()
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

max_epochs = 15
for epoch in range(max_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for it, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        optimizer.step()

        pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}")

    # eval:
    model.eval()
    with torch.no_grad():
        context = "O God, O God!"
        x = torch.tensor(
            [train_dataset.stoi[s] for s in re.split(r"\b", context)], dtype=torch.long
        )[None, ...].to(device)
        y = sample(model, x, 200, temperature=1.0, sample=True, top_k=10)[0]
        completion = "".join([train_dataset.itos[int(i)] for i in y])
        print(completion)

    # save model
    print("saving model")
    ckpt_path = os.path.join(os.getcwd(), "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    model.train()
