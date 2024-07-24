import csv
import clevercsv
import pandas as pd
import time
from asyncio import TimeoutError

class TURL_embedding_generator:
    def __init__(self, data_dir: str="experiments/ablation_modified/code_TURL/datasets/turl_datasets/",
                 model_checkpoint_path: str="experiments/ablation_modified/code_TURL/models/turl_pretrained.bin",
                 max_lines: int=128, config_name: str="experiments/ablation_modified/code_TURL/TURL/configs/table-base-config_v2.json",
                 sampling_size: int=10) -> None:
        #Model variables
        self.max_lines = max_lines
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
        self.entity_wikid2id = {self.entity_vocab[x]['wiki_id']:x for x in self.entity_vocab}
        self.sampling_method = 'random'

        self.config = TableConfig.from_pretrained(config_name)
        self.config.output_attentions = True

        self.model = HybridTableMaskedLM(self.config, is_simple=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model_checkpoint = torch.load(model_checkpoint_path)
        self.model.load_state_dict(model_checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.dataset = WikiHybridTableDataset(data_dir,self.entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150,
        src="dev", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0)
        self.text_to_entity = get_text_to_entity_dict(self.entity_vocab)
        self.sampling_size = sampling_size
        
        #Dataset variables
        self.src = 'train'
        self.mode = 0
        self.max_cell = float(100)
        self.max_title_length = 50

        self.max_header_length = 10
        self.max_cell_length = 10
        self.force_new = False
        self.max_input_tok = 350
        self.max_input_ent = 150

    def prepare_dataframe(self, table: str, max_lines: int) -> pd.DataFrame:
        
        #table = pd.read_csv(table, on_bad_lines='skip', encoding_errors='ignore', quoting=csv.QUOTE_NONE, dtype=str, lineterminator='\n', header=None)
        try:
            table = clevercsv.read_dataframe(table)
        except Exception as e:
            if isinstance(e, TimeoutError):
                raise TimeoutError
            table = pd.read_csv(table, on_bad_lines='skip', encoding_errors='ignore', quoting=csv.QUOTE_NONE, dtype=str, lineterminator='\n', header=None)
            #print(f'Table loaded using pd.read_csv')
        return table.iloc[:max_lines, :max_lines]

    def get_embedding(self, table, save_only_centroid: bool=True, cpu: bool=False):
        pgEnt = '[PAD]'
        pgTitle = ''
        secTitle = ''
        caption = ''
        headers = list(table.columns)
        table = table
        core_entities = [] # This will be the subject column entities
        core_entities_text = []
        all_entity_cand = []
        if self.sampling_method == 'head':
            table_subset = table.head(self.sampling_size) # first 10 rows  
        elif self.sampling_method == 'random':
            if self.sampling_size == -1:
                table_subset = table
            else:
                try:
                    table_subset = table.sample(n=self.sampling_size, random_state=1) # random 10 rows
                except Exception as e:
                    if isinstance(e, TimeoutError):
                        raise TimeoutError
                    table_subset = table
        elif self.sampling_method == 'diverse':
            table_subset = select_k_diverse_row(table, self.sampling_size) # diversed row sampling with 10 rows
        
        # print('Candidate entity extraction starts')
        start = time.time()
        for index, row in table_subset.iterrows():
            for columnIndex, value in row.items():
                entity = text_preprocessing(str(value).replace(' ', '_'))
                if entity in self.text_to_entity:
                    core_entities.append(self.text_to_entity[entity])
                    core_entities_text.append(entity) 
                    all_entity_cand.append(self.text_to_entity[entity])
                else:
                    sub_entities = entity.split('_')
                    if sub_entities != None:
                        for sub_entity in sub_entities:
                            if sub_entity in self.text_to_entity:
                                core_entities.append(self.text_to_entity[sub_entity])
                                core_entities_text.append(sub_entity) 
                                all_entity_cand.append(self.text_to_entity[sub_entity])
        all_entity_cand = list(set(all_entity_cand))
        
        end = time.time()
        #print(f'Candidate entity extraction ends {end-start}s. Found {len(all_entity_cand)} entities')

        #print('Input building starts')
        start = time.time()
        input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, \
                candidate_entity_set = CF_build_input1(pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, all_entity_cand, self.dataset)
        end = time.time()
        #print(f'Input building ends {end-start}s')

        if cpu:
            self.model.cpu()
        else:
            input_tok = input_tok.to(self.device)
            input_tok_type = input_tok_type.to(self.device)
            input_tok_pos = input_tok_pos.to(self.device)
            input_tok_mask = input_tok_mask.to(self.device)
            input_ent_text = input_ent_text.to(self.device)
            input_ent_text_length = input_ent_text_length.to(self.device)
            input_ent = input_ent.to(self.device)
            input_ent_type = input_ent_type.to(self.device)
            input_ent_mask_type = input_ent_mask_type.to(self.device)
            input_ent_mask = input_ent_mask.to(self.device)
            candidate_entity_set = candidate_entity_set.to(self.device)

        # print('Embedding generation starts')
        start = time.time()
        with torch.no_grad():
            tok_outputs, ent_outputs = self.model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                            input_ent_text, input_ent_text_length, input_ent_mask_type,
                            input_ent, input_ent_type, input_ent_mask, candidate_entity_set)
        end = time.time()
        # print(f'Embedding generation ends {end-start}s')

        if save_only_centroid:
            table_mean_rep = {}
            table_mean_rep['entities_only'] = torch.mean(ent_outputs[1][0], 0, dtype=torch.float)
            table_mean_rep['metadata_only'] = torch.mean(tok_outputs[1][0], 0, dtype=torch.float)
            table_comb = torch.cat((tok_outputs[1][0], ent_outputs[1][0]), dim=0)
            table_mean_rep['metadata_entities'] = torch.mean(table_comb, 0, dtype=torch.float)
            table_repr = table_mean_rep
        else:
            table_repr = tok_outputs, ent_outputs
        if cpu:
            self.model.to(self.device)
        return table_repr

    def embed(self, file: str):
        # print('Preparing dataframe')
        start = time.time()
        df = self.prepare_dataframe(file, self.max_lines)
        end = time.time()
        # print(f'Dataframe prepared {end-start}s')
        # print('Embedding generation starts')
        start = time.time()
        try:
            embeddings = self.get_embedding(df, cpu=False)['metadata_entities']
        except Exception as e:
            if isinstance(e, TimeoutError):
                raise TimeoutError
            print(f'Cuda out of memory trying to comput the embedding of {file} on cpu')
            embeddings = self.get_embedding(df, cpu=True)['metadata_entities']

        end = time.time()
        # print(f'Embedding generated {end-start}s')

        embeddings = embeddings.reshape(-1)
        return {"file_embedding":embeddings.cpu()}
