from dataset import *

def get_dataset(datasetname, seq_len=128):
    if datasetname == 'Twitter':
        in_data_path = '../data/in-domain/Twittergraph'
        in_twitter_ids = np.load('../data/twitter_in_ids.npy')
        out_data_path = '../data/out-of-domain/Twittergraph'
        out_twitter_ids = np.load('../data/twitter_out_ids.npy')
        in_dataset = UCDDataset(in_twitter_ids, in_data_path, seq_len)
        out_dataset = UCDDataset(out_twitter_ids, out_data_path, seq_len)
    if datasetname == 'Weibo':
        in_data_path = '../data/in-domain/Weibograph'
        in_weibo_ids = np.load('../data/weibo_in_ids.npy')
        out_data_path = '../data/out-of-domain/Weibograph'
        out_weibo_ids = np.load('../data/weibo_out_ids.npy')
        in_dataset = UCDDataset(in_weibo_ids, in_data_path, seq_len)
        out_dataset = UCDDataset(out_weibo_ids, out_data_path, seq_len)
    
    return in_dataset, out_dataset