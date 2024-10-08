import numpy as np
import torch
from transformers import BertTokenizer, BertModel, logging
from Bio import SeqIO
import time
import pandas as pd
import tqdm
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
    assert "sequences" in species_dict, "dictionary doesn't contain 'sequences'"
    logging.set_verbosity_error()

    # record to gene name, if False --> protein sequence is skipped
    sequence_path = species_dict["sequences"]

    if "embeddings" in species_dict:
        embedding_path = species_dict["embeddings"]
    else:
        embedding_path = sequence_path.split(".")[0] + "_embeddings.pkl"
        species_dict["embeddings"] = embedding_path

    model = BertModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False, padding=True, max_length=max_length,
                                              truncation="longest_first")

    records = list(SeqIO.parse(sequence_path, "fasta"))
    n_records = len(records)
    sequences = [str(record.seq)[:max_length] for record in records]
    lengths = [len(sequence) for sequence in sequences]
    names = list(map(protein_to_gene, records))

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


def calc_dist(species_dict_A, species_dict_B, match_path, threshold=0.005):
    assert "embeddings" in species_dict_A, "dictionary doesn't contain 'embeddings'"
    assert "embeddings" in species_dict_B, "dictionary doesn't contain 'embeddings'"

    emb_df_A = pd.read_pickle(species_dict_A["embeddings"])
    emb_df_B = pd.read_pickle(species_dict_B["embeddings"])

    dist_matrix = scipy.spatial.distance.cdist(emb_df_A.to_numpy(), emb_df_B.to_numpy(), metric="cosine")

    dist_matrix = 1 - dist_matrix
    dist_matrix[dist_matrix <= 1 - threshold] = 0
    dist_matrix = scipy.sparse.coo_matrix(dist_matrix)

    with open(match_path, "wb") as f:
        pickle.dump([emb_df_A.index.values, emb_df_B.index.values, dist_matrix], f)

    print("gene matches saved to", match_path)


def calc_dist_blast(A_to_B_path, B_to_A_path, match_path, threshold=1e-6):
    n1n2 = pd.read_csv(A_to_B_path, sep="\t", header=None)
    n2n1 = pd.read_csv(B_to_A_path, sep="\t", header=None)
    data = {}

    for i, entry in n1n2.iterrows():
        gene_A = entry[0]
        gene_B = entry[1]
        value = entry[11]
        if entry[10] < threshold:
            data[(gene_A, gene_B)] = value

    for i, entry in n2n1.iterrows():
        gene_B = entry[0]
        gene_A = entry[1]
        value = entry[11]
        if entry[10] < threshold:
            if (gene_A, gene_B) in data:
                data[(gene_A, gene_B)] = (data[(gene_A, gene_B)] + value) / 2
            else:
                data[(gene_A, gene_B)] = value

    keys, items = data.keys(), data.values()
    row, col = map(list, zip(*keys))

    genes_A = sorted(list(set(row)))
    genes_B = sorted(list(set(col)))

    row = [genes_A.index(x) for x in row]
    col = [genes_B.index(x) for x in col]

    dist_matrix = scipy.sparse.coo_matrix((list(items), (row, col)), shape=(len(genes_A), len(genes_B)), dtype=float)

    with open(match_path, "wb") as f:
        pickle.dump([genes_A, genes_B, dist_matrix], f)

def calc_dist_blast11(A_to_B_path, B_to_A_path, match_path, threshold=1e-6):
    n1n2 = pd.read_csv(A_to_B_path, sep="\t", header=None)
    n2n1 = pd.read_csv(B_to_A_path, sep="\t", header=None)
    n1n2 = n1n2[n1n2[10] < threshold]
    n2n1 = n2n1[n2n1[10] < threshold]

    n1n2 = n1n2.groupby(0).head(1)
    n1n2 = n1n2.set_index(0)
    n2n1 = n2n1.groupby(0).head(1)
    n2n1 = n2n1.set_index(0)

    blast_11 = n1n2[n1n2[1].isin(n2n1.index)]
    blast_11 = blast_11[blast_11.index == n2n1.loc[blast_11[1]][1]]
    blast_11 = blast_11.sort_index()

    genes_A = list(blast_11.index)
    genes_B = list(blast_11[1].values)

    dist_matrix = scipy.sparse.coo_matrix(np.diag(np.ones(len(genes_A))))

    with open(match_path, "wb") as f:
        pickle.dump([genes_A, genes_B, dist_matrix], f)
