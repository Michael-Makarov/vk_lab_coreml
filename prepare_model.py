import argparse
from model_utils import *
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Train and save the model')
    parser.add_argument('--data-path', required=True, help="path to dataset")
    parser.add_argument('--model-path', required=True, help="where to save the model")
    parser.add_argument('--calc-als', type=int, default=1, help="whether to calculate als metrics")
    parser.add_argument('--use-gridsearch', type=int, default=1, help="whether to use gridsearch")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    print('Reading data')
    all_data = pd.read_csv(args.data_path)
    
    data_train, data_test = split_data(all_data)
    if args.calc_als:
        vals = get_als_values(all_data, data_train, data_test, args.use_gridsearch)
        print("ALS metrics")
        print("NDCG@10:", vals[0])
        print("NDCG@1:", vals[1])
        
    users, seqs = get_sequences(all_data)
    emb_model = Word2Vec(seqs, size=100, window=5, min_count=0, sg=1, hs=0, negative=5, workers=3, iter=5)
    user_mapping = get_user_mapping(all_data)
    similar_cnts = [3, 5, 7] if args.use_gridsearch else [5]
    best_model = None
    best_res = [0, 0]
    for cnt in similar_cnts:
        mapping, reverse_mapping, embedding, most_similar = get_embedding(emb_model, all_data, cnt)
        model = WRMFEmbedded(user_mapping, mapping, reverse_mapping, embedding, most_similar, size=100, alpha=0.1, lambd=0.01, cuda=True)
        prep_data = model.prepare_data(data_train)
        model.fit(prep_data, epochs=4, verbose=0, batch_size=1024, lr=0.5)
        m1 = calc_mean_ndcg_wrmf(model, data_test, 1)
        m10 = calc_mean_ndcg_wrmf(model, data_test, 10)
        if best_res[0] < m10:
            best_res = [m10, m1]
            best_model = model
    print("Model metrics")
    print("NDCG@10:", best_res[0])
    print("NDCG@1:", best_res[1])
    torch.save(best_model, args.model_path)
