import polars as pl
import numpy as np

from os import scandir

metadata = pl.scan_csv("validated.tsv", separator="\t")
labels = metadata.select("path", pl.col("gender").str.contains("female").alias("label")).drop_nulls().collect()
pairs = ((ent.name, np.load(ent.path)) for ent in scandir("tokens"))
paths, tokens = zip(*pairs)
tokens = pl.DataFrame({"path": paths, "tokens": tokens}).with_columns(pl.col("path").str.replace(".npy$", ""))

df = labels.join(tokens, on="path")
df.write_parquet("tokens.parquet")

pass
