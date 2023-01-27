import torch
from transformers import BertTokenizer, BertModel, pipeline
from Bio import SeqIO
import time
import pandas as pd
import tqdm
from transformers import logging
import scipy
import pickle


def embed_proteins(species_dict,
                   model_path="Rostlab/prot_bert",
                   max_length=2500,  # truncate sequences longer than max_length amino acids
                   max_aa=5000,  # max amino acids in a batch
                   max_batch=100,  # max sequences in a batch
                   protein_to_gene=lambda x: x.id  # [Bio.SeqRecord --> str] map record to gene name,
                   # or False to skip record
                   ):
    assert "sequence_path" in species_dict, "dictionary doesn't contain 'sequence_path'"
    logging.set_verbosity_error()

    # record to gene name, if False --> protein sequence is skipped
    sequence_path = species_dict["sequence_path"]

    if "embedding_path" in species_dict:
        embedding_path = species_dict["embedding_path"]
    else:
        embedding_path = sequence_path.split(".")[0] + "_embeddings.pkl"
        species_dict["embedding_path"] = embedding_path

    model = BertModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False, padding=True, max_length=max_length,
                                              truncation="longest_first")

    records = list(SeqIO.parse(sequence_path, "fasta"))[:100]
    n_records = len(records)
    sequences = [str(record.seq)[:max_length] for record in records]
    lengths = [len(sequence) for sequence in sequences]
    names = list(map(protein_to_gene, records))
    for i in range(10, 20):
        names[i] = False

    names, sequences = map(list, zip(*sorted(filter(lambda x: x[0] is not False, zip(names, sequences)),
                                             key=lambda x: len(x[1]), reverse=True)))
    print(f"calculating ProtBERT embeddings for {len(names)}/{len(records)} sequences from {sequence_path}")

    batch = list()
    embeddings = []
    embedding_names = []
    failed_names = []
    start = time.time()
    for i, (name, seq) in enumerate(tqdm.tqdm(zip(names, sequences), total=len(sequences))):
        if name in embedding_names:  # skip duplicates
            continue
        len_seq = len(seq)
        seq = ' '.join(list(seq))
        batch.append((name, seq, len_seq))
        if i == len(sequences) - 1 or len(batch) >= max_batch or sum(x for _, _, x in batch) + len_seq > max_aa:
            batch_names, batch_seqs, batch_len_seqs = map(list, zip(*batch))
            token_encoding = tokenizer.batch_encode_plus(batch_seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids'])
            attention_mask = torch.tensor(token_encoding['attention_mask'])
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                print(e)
                print(f"RuntimeError during embedding for {name} (L={len(seq)})")
                print(batch_names)
                print(batch_len_seqs)
                failed_names += batch_names
                batch = list()
                continue
            for j, s_len in enumerate(batch_len_seqs):  # for each protein in the current mini-batch
                emb = embedding_repr.last_hidden_state[j, :s_len]
                protein_emb = emb.mean(dim=0)
                if torch.cuda.is_available():
                    embeddings.append(protein_emb.detach().cpu().numpy().squeeze())
                else:
                    embeddings.append(protein_emb.detach().numpy().squeeze())
                embedding_names.append(batch_names[j])
            batch = list()

    df = pd.DataFrame(embeddings, columns=pd.RangeIndex(0, 1024, 1), index=embedding_names)
    df.to_pickle(embedding_path)


def match_genes(species_dict_A, species_dict_B, match_path, threshold=0.005):
    assert "embedding_path" in species_dict_A, "dictionary doesn't contain 'embedding_path'"
    assert "embedding_path" in species_dict_B, "dictionary doesn't contain 'embedding_path'"

    emb_df_A = pd.read_pickle(species_dict_A["embedding_path"])
    emb_df_B = pd.read_pickle(species_dict_B["embedding_path"])

    dist_matrix = scipy.spatial.distance.cdist(emb_df_A.to_numpy(), emb_df_B.to_numpy(), metric="cosine")

    dist_matrix = 1 - dist_matrix
    dist_matrix[dist_matrix <= 1 - threshold] = 0
    dist_matrix = scipy.sparse.coo_matrix(dist_matrix)

    with open(match_path, "wb") as f:
        pickle.dump([emb_df_A.index.values, emb_df_B.index.values, dist_matrix], f)

    print("gene matches saved to", match_path)
