from turl import TURL_embedding_generator
import pandas as pd
from tqdm import tqdm
from interruptingcow import timeout
from asyncio import TimeoutError
import pickle

data_path = '/home/magritte/magritte/data/survey/'
csv_path = '/home/magritte/magritte/data/survey/csv/'
out_dir = '/home/magritte/magritte/tmp/magritte_ablation/turl_embeddings_tmp/turl_embeddings_additional_5min.pkl'
previous_output = '/home/magritte/magritte/tmp/magritte_ablation/turl_embeddings_tmp/turl_embeddings.pkl'
max_t_exec = 300

if __name__ == '__main__':
    if previous_output == None:
        train = pd.read_csv(data_path+'train.csv')
        test = pd.read_csv(data_path+'test.csv')
        dev = pd.read_csv(data_path+'dev.csv')
        file_list = list(train['filename']) + list(test['filename']) +list(dev['filename'])
    else: 
        with open(previous_output, 'rb') as handle:
            b = pickle.load(handle)
            file_list = b['overtime']

    embedding_dictionary = {'overtime':[], 'other_errors':[],'embeddings':{}}
    model = TURL_embedding_generator(sampling_size=-1)
    print('Embedding generation starting')
    
    for file in tqdm(file_list):
        try:
            try:
                with timeout(max_t_exec, exception=TimeoutError):
                    emb = model.embed(csv_path+file)
                    embedding_dictionary['embeddings'][file] = emb['file_embedding']
            except TimeoutError:
                print(f'Overtime in table {file}')
                model.model.to(model.device)
                embedding_dictionary['overtime'].append(file)
        except:
            print(f'Generic error in table {file}')
            model.model.to(model.device)
            embedding_dictionary['other_errors'].append(file)
    with open(out_dir, 'wb') as f:
        pickle.dump(embedding_dictionary, f)
    
    print('End')