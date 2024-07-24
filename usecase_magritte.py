import os
import pandas as pd
from magritte import MaGRiTTE

dataset_path = "data/massbay/"
tables = []
for f in os.listdir(dataset_path):
    table = MaGRiTTE.extract_table(filepath=os.path.join(dataset_path, f))
    table = MaGRiTTE.delete_metadata(table)
    table = MaGRiTTE.disambiguate_column_headers(table)
    tables += [table]
    print(table)
    
union_table = pd.concat(tables).reset_index(drop=True)
print(union_table)
union_table.to_csv("results/massbay/magritte_integrated.csv", index=False)
