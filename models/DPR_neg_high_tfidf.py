import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel, AdamW
import numpy as np
import shutil
from tqdm import tqdm
from configparser import ConfigParser
import os
import json
import argparse
import shutil
import random

def load_data(data_dir):

    df = {}
    df["train"] = pd.read_pickle(os.path.join(data_dir, "train_w_review_sents.pkl"))
    df["valid"] = pd.read_pickle(os.path.join(data_dir, "valid_w_review_sents.pkl"))
    df["test"] = pd.read_pickle(os.path.join(data_dir, "test_w_review_sents.pkl"))

    claim_ids = {}
    claim_txt = {}
    sentences = {}
    sent_ids = {}
    id2sents = {}
    id2claims = {}

    for partition in df.keys():
        claim_ids[partition] = []
        claim_txt[partition] = []
        sentences[partition] = []
        sent_ids[partition] = []
        id2sents[partition] = {}
        id2claims[partition] = {}

        for _, row in df[partition].iterrows():
            claim_ids[partition].append(row["id"])
            claim_txt[partition].append(row["text"])
            sentences[partition].extend(row["review_sentences"])
            sent_ids[partition].extend([row["id"]] * len(row["review_sentences"]))
            id2sents[partition][row["id"]] = row["review_sentences"]
            id2claims[partition][row["id"]] = row["text"]
    
    nht_file = os.path.join(data_dir, "neg_high_tfidf.json")
    with open(nht_file) as f_obj:
        neg_high_tfidf = json.load(f_obj)
    
    neg_high_tfidf = {int(key): val for key, val in neg_high_tfidf.items()}

    return claim_ids, claim_txt, sentences , sent_ids, id2claims, id2sents, neg_high_tfidf


def build_validation_data_loader(
    claim_ids, claim_txt, sentences , sent_ids, claim_msl, sent_msl, tokenizer
):

  partitions = ["valid", "test"]

  ct_input_ids = {}
  ct_dataset = {}
  ct_dataloader = {}
  for partition in partitions:
    ct_input_ids[partition] = tokenizer(claim_txt[partition], padding="max_length", truncation=True, max_length=claim_msl, return_tensors="pt").input_ids
    ct_dataset[partition] = TensorDataset(ct_input_ids[partition])
    ct_dataloader[partition] = DataLoader(ct_dataset[partition], shuffle=False, batch_size=64)

  sent_input_ids = {}
  sent_dataset = {}
  sent_dataloader = {}
  for partition in partitions:
    sent_input_ids[partition] = tokenizer(sentences[partition], padding="max_length", truncation=True, max_length=sent_msl, return_tensors="pt").input_ids
    sent_dataset[partition] = TensorDataset(sent_input_ids[partition])
    sent_dataloader[partition] = DataLoader(sent_dataset[partition], shuffle=False, batch_size=64)

  return ct_dataloader, sent_dataloader


def convert_to_tensors(
    train_id2claims,
    train_id2sents,
    tokenizer,
    claim_msl,
    sent_msl
):
  train_id2claims_tensor = {}
  train_id2sents_tensor = {}
  for id in train_id2claims.keys():
    if len(train_id2sents[id]) == 0:
      continue
    train_id2claims_tensor[id] = tokenizer(train_id2claims[id], padding="max_length", truncation=True, max_length=claim_msl, return_tensors="pt").input_ids
    train_id2sents_tensor[id] = tokenizer(train_id2sents[id], padding="max_length", truncation=True, max_length=sent_msl, return_tensors="pt").input_ids

  return train_id2claims_tensor, train_id2sents_tensor


def compute_recall(claim_vecs, sentence_vecs, claim_labels, sentence_labels, top_k_list):

    sentence_labels = np.array(sentence_labels)
    recalls = {top_k: list()  for top_k in top_k_list}
    for index in range(claim_vecs.shape[0]):
        claim_vec = claim_vecs[index]
        claim_label = claim_labels[index]

        sim_scores = sentence_vecs.dot(claim_vec)

        label_sent_indices = np.where(sentence_labels==claim_label)[0]

        sorted_indices = np.argsort(sim_scores)[::-1]

        total_relevant = len(label_sent_indices)
        for top_k in top_k_list:
            retrieved_relevant = len(np.intersect1d(sorted_indices[:top_k], label_sent_indices, assume_unique=True))
            recall = retrieved_relevant / total_relevant

            recalls[top_k].append(recall)

    return recalls

class SentenceSimilarityModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.c_encoder = RobertaModel.from_pretrained('distilroberta-base')
    self.s_encoder = RobertaModel.from_pretrained('distilroberta-base')

  def embed_claims(self, claims):
    return self.c_encoder(claims).pooler_output

  def embed_sents(self, sents):
    return self.s_encoder(sents).pooler_output

  def forward(self, claims, sents):

    c_embeddings = self.c_encoder(claims).pooler_output
    s_embeddings = self.s_encoder(sents).pooler_output

    s_embeddings_t = s_embeddings.transpose(0, 1)

    sim_scores = torch.matmul(c_embeddings, s_embeddings_t)
    return c_embeddings, s_embeddings, sim_scores

def biencoder_nll_loss(sim_scores, labels):

  softmax_scores = F.log_softmax(sim_scores, dim=1)
  loss = F.nll_loss(softmax_scores, labels, reduction="none")

  max_score, max_idxs = torch.max(softmax_scores, 1)
  correct_predictions_count = (max_idxs == labels).sum()

  return loss, correct_predictions_count

def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def load_model_checkpoint(model_checkpoint_file):
    if not os.path.isfile(model_checkpoint_file):
        return None
    
    return torch.load(model_checkpoint_file)


def main(args):
    claim_ids, claim_txt, sentences , sent_ids, id2claims, id2sents, neg_high_tfidf = load_data(args.data_dir)

    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    claim_msl = 120
    sent_msl = 320

    eval_ct_dataloader, eval_sent_dataloader = build_validation_data_loader(
        claim_ids, claim_txt, sentences , sent_ids, claim_msl, sent_msl, tokenizer
    )

    train_id2claims_tensor, train_id2sents_tensor = convert_to_tensors(
        id2claims["train"],
        id2sents["train"],
        tokenizer,
        claim_msl,
        sent_msl
    )
    train_dataset = TensorDataset(torch.tensor(list(train_id2claims_tensor.keys())).cuda())
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    model = SentenceSimilarityModel().cuda()
    optimizer = get_optimizer(model)
    
    model_checkpoint_file = os.path.join(args.checkpoint_dir, "model.tar")
    temp_checkpoint_file = os.path.join(args.checkpoint_dir, "temp_model.tar")
    
    model_checkpoint = load_model_checkpoint(model_checkpoint_file)
    
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint["model_state_dict"])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        start_epoch = model_checkpoint["epoch"] + 1
        print(f"Resuming model training from epoch: {start_epoch+1}")
    else:
        start_epoch = 0

    
    output_dir = os.path.join(args.output_dir, f"DPR_nht_batch_size_{args.batch_size}_exp_{args.exp_num}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    log_every = 25
    validate_every = 10

    for epoch in range(start_epoch, 500):
        train_loss = []
        train_correct = 0
        train_total = 0
        model.train()
        with torch.set_grad_enabled(True):
            for index, batch in enumerate(train_dataloader):
                indices = batch[0]
                indices_set = set(indices.tolist())

                claims = []
                sents = []
                neg_sents = []
                for id in indices:
                    id = id.item()
                    claims.append(train_id2claims_tensor[id].squeeze())
                    sents.append(train_id2sents_tensor[id][epoch%train_id2sents_tensor[id].size(0)])
                    
                    nht_list = neg_high_tfidf[id]
                    nht_item = random.choice(nht_list)
                    neg_sents.append(train_id2sents_tensor[nht_item[1]][nht_item[2]])

                claims = torch.stack(claims).cuda()
                sents = torch.stack(sents).cuda()
                neg_sents = torch.stack(neg_sents).cuda()

                c_embeddings, s_embeddings, sim_scores = model(claims, sents)
                neg_s_embeddings = model.embed_sents(neg_sents)
                neg_sim_scores = torch.mul(c_embeddings, neg_s_embeddings).sum(-1).unsqueeze(dim=1)
                scores = torch.cat((sim_scores, neg_sim_scores), dim=1)
                labels = torch.arange(len(indices)).cuda()

                loss, correct_predictions_count = biencoder_nll_loss(scores, labels)

                train_correct += correct_predictions_count
                train_total += len(indices)

                train_loss.extend(loss.detach().cpu().tolist())

                batch_loss = torch.mean(loss)

                if ((index + 1) % log_every) == 0:
                    print(f"Epoch: {epoch+1} Batch# {index+1}, Batch loss: {batch_loss.item():.2f} Correct Predictions: {correct_predictions_count}")

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            print()
            print(f"Epoch # {epoch+1} Train Loss: {np.mean(train_loss).item():.2f} Correct Predictions: {(train_correct/train_total)*100:.2f}")
            print()
            
            
            try:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, 
                    temp_checkpoint_file
                )
                save_successful = True
            except:
                print(f"Warning! Could not save model for epoch: {epoch+1}!")
                save_successful = True
            
            if save_successful:
                shutil.move(temp_checkpoint_file, model_checkpoint_file)
                

            if ((epoch+ 1) % validate_every) == 0:
                print()
                print("Validating model.")
                model.eval()
                with torch.no_grad():
                    ct_vecs = {}
                    for partition in eval_ct_dataloader.keys():
                        ct_vecs[partition] = []
                        for batch in tqdm(eval_ct_dataloader[partition]):
                            ct_batch = batch[0].cuda()
                            embeddings = model.embed_claims(ct_batch)
                            ct_vecs[partition].append(embeddings.cpu())

                        ct_vecs[partition] = torch.cat(ct_vecs[partition]).numpy()

                    sent_vecs = {}
                    for partition in eval_sent_dataloader.keys():
                        sent_vecs[partition] = []
                        for batch in tqdm(eval_sent_dataloader[partition]):
                            sent_batch = batch[0].cuda()
                            embeddings = model.embed_sents(sent_batch)
                            sent_vecs[partition].append(embeddings.cpu())

                        sent_vecs[partition] = torch.cat(sent_vecs[partition]).numpy()

                top_k_list = [10, 25, 50, 100]
                eval_results = {}
                for partition in eval_ct_dataloader.keys():
                    print(partition)
                    recalls = compute_recall(ct_vecs[partition], sent_vecs[partition], claim_ids[partition], sent_ids[partition], top_k_list)
                    eval_results[partition] = {}
                    for top_k, recall in recalls.items():
                        eval_results[partition][top_k] = round(np.mean(recall),2)*100
                        print(f"Top-{top_k} Recall: {round(np.mean(recall),2)*100}")
                    
                print()
                eval_results_filename = f"eval_results_{epoch+1}.json"
                eval_results_file_path = os.path.join(output_dir, eval_results_filename)
                print(f"Saving eval results to {eval_results_file_path}")
                with open(eval_results_file_path, "w") as f_obj:
                    json.dump(eval_results, f_obj, indent=4)
                model_filename = f"model_{epoch+1}.pt"
                model_file_path = os.path.join(output_dir, model_filename)
                print(f"Saving model to {model_file_path}")
                torch.save(model, model_file_path)
                print()
                



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Dense Passage Retrieval Model Training & Evaluation')
    
    parser.add_argument('--exp_num', type=str, help='Please specify the experiment number')
    parser.add_argument('--batch_size', type=int, help='Please specify the batch size')
    parser.add_argument('--checkpoint_dir', type=str, help='Please specify the checkpoint directory')
    parser.add_argument('--output_dir', type=str, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, help='Please specify the data directory')
    
    args = parser.parse_args()
    
    main(args)
