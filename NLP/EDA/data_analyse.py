# %%
import pandas as pd

main_data = pd.read_csv('../../dataset/TED-Speech-data/ted_main.csv')
scripts = pd.read_csv('../../dataset/TED-Speech-data/transcripts.csv')


# %%
from matplotlib import pyplot as plt

# %%
scripts['new'] = scripts[['transcript']]
# %%
scripts['slen'] = scripts[['transcript']]
# %%
