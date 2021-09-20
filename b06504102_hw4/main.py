from tqdm import tqdm
from tqdm.notebook import tqdm
import torch,json,csv
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from Mynet import Classifier, model_fn
from myDataset import myDataset, InferenceDataset, get_dataloader, inference_collate_batch
from LRScheduler import get_cosine_schedule_with_warmup

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_parse_args():
  """arguments"""
  config = {
    "data_dir": "./Dataset",
    "save_path": "model_0.ckpt",
    "batch_size": 32,
    "n_workers": 8,
    "valid_steps": 2000,  # 每多少圈 valid 一次
    "warmup_steps": 30000, # Schduler 
    "save_steps": 10000,  # 每多少圈 save 一次"最好的"
    "total_steps": 2000000, # 總共跑幾圈
  }

  return config

def inference_parse_args():
  """arguments"""
  config = {
    "data_dir": "./Dataset",
    "model_path": "./model_0.ckpt",
    "output_path": "./output_12.csv",
  }

  return config

def Train(
  data_dir,
  save_path,
  batch_size,
  n_workers,
  valid_steps,
  warmup_steps,
  total_steps,
  save_steps,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  model = Classifier(n_spks=speaker_num).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!",flush = True)

  best_accuracy = -1.0
  best_state_dict = None
  valid_accuracy = 0.0
  

  for step in range(total_steps):
    # Get data
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    loss, accuracy = model_fn(batch, model, criterion, device)
    batch_loss = loss.item()
    batch_accuracy = accuracy.item()

    # Updata model
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # Do validation
    if (step + 1) % valid_steps == 0:

      valid_accuracy = valid(valid_loader, model, criterion, device)

      # keep the best model
      if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_state_dict = model.state_dict()
        print(f"Step: {step+1:05d}/{total_steps:05d}, Best: {best_accuracy:.5f}")
      else:
        print(f"Step: {step+1:05d}/{total_steps:05d}, Train: {batch_accuracy:.5f}, Valid: {valid_accuracy:.5f}")


    # Save the best model so far.
    if (step + 1) % save_steps == 0 and best_state_dict is not None:
      torch.save(best_state_dict, save_path)
      



#   pbar.close()

def valid(dataloader, model, criterion, device): 
  """Validate on validation set."""

  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()

  model.train()

  return running_accuracy / len(dataloader)

def Inference(
  data_dir,
  model_path,
  output_path,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  mapping_path = Path(data_dir) / "mapping.json"
  mapping = json.load(mapping_path.open())

  dataset = InferenceDataset(data_dir)
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=inference_collate_batch,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  speaker_num = len(mapping["id2speaker"])
  model = Classifier(n_spks=speaker_num).to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)

  results = [["Id", "Category"]]
  for feat_paths, mels in dataloader:
    with torch.no_grad():
      mels = mels.to(device)
      outs = model(mels)
      preds = outs.argmax(1).cpu().numpy()
      for feat_path, pred in zip(feat_paths, preds):
        results.append([feat_path, mapping["id2speaker"][str(pred)]])
  
  with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)


same_seeds(0)
Train(**train_parse_args())
Inference(**inference_parse_args())
