import polars as pl
import pandas as pd
from datetime import timedelta
from typing import Iterable, List, Dict


def read_segments(segments: Iterable) -> pl.DataFrame:
    data = pd.DataFrame(segments)
    words = data.words.explode().apply(pd.Series)
    words.columns = ['start', 'end', 'word', 'prob']
    words = pl.DataFrame(words).lazy()\
        .with_columns(pl.col("word").str.ends_with(".").cast(pl.Int64).alias("is_end")) \
        .with_columns(pl.col("is_end").shift_and_fill(1, 0)) \
        .with_columns(pl.col("is_end").cumsum()) \
        .groupby("is_end") \
        .agg(
        pl.col("word").apply(lambda x: "".join(x)).alias("text"),
        pl.col("start").min().alias("start"),
        pl.col("end").max().alias("end")
    ).with_columns(pl.col("text").str.slice(1)).sort(by="is_end").collect()
    text = "".join(data.text.to_list()).replace("  ", " ").split(". ")
    print(words.shape, "  ", len(text))
#   assert words.shape[0] == len(text)
    return words

# generate srt file format data


def parse_srt(data: pl.DataFrame) -> pl.DataFrame:
    # print(json.dumps(pip_res))
    # data = pl.read_json(io.StringIO(json.dumps(pip_res)))
    # data = data.with_columns(pl.col("timestamp").arr.get(0).alias("start"), pl.col("timestamp").arr.get(1).alias("end"))
    data = data.with_columns(
        (pl.col(["start", "end"]).apply(lambda x: str(timedelta(seconds=x)))).suffix("_format")) \
        .select(
            (pl.col("is_end") + 1).cast(pl.Utf8),
            pl.concat_str([
                pl.col("start_format").str.slice(
                    0, length=11).str.replace(".", ",", literal=True),
                pl.col("end_format").str.slice(
                    0, length=11).str.replace(".", ",", literal=True)
            ], separator=" --> ").alias("timestamp"),
            pl.col("text"),
    )
    return data

# generate request file


def gen_request_file(data: pl.DataFrame, prompt_q: str, prompt_a: str) -> pl.DataFrame:
    return data.lazy().with_columns(pl.col("is_end").cast(pl.Int64)) \
        .with_columns((((pl.col("is_end")-1) // 10)).alias("group")) \
        .with_columns(pl.concat_str([pl.col("is_end").cast(pl.Utf8),
                                    pl.lit(". "),
                                    pl.col("text")]).alias("text")) \
        .groupby("group", maintain_order=True) \
        .agg(
        pl.col("text").apply(lambda x: "".join(x)),
        pl.count()
    ) \
        .select(pl.concat_str([
            pl.lit("you should translate the belowing "),
            pl.col("count").cast(pl.Utf8),
            pl.lit(" lines to chinese. Important: You should return exactly"),
            pl.col("count").cast(pl.Utf8),
            pl.lit(" Chinese sentences end with '。',\n"),
            pl.col("text")]).alias("content2")
    ) \
        .with_columns(
        pl.lit("system").alias("role"),
        pl.lit("You are a translate machine that translate Formula 1 content from English to Chinese. Please preserve the number before each sentence.").alias("content"),
        pl.lit("user").alias("role0"),
        pl.lit(prompt_q).alias("content0"),
        pl.lit("assistant").alias("role1"),
        pl.lit(prompt_a).alias("content1"),
        pl.lit("user").alias("role2")) \
        .select(
        pl.struct(["role", "content"]).alias("zero"),
        pl.struct(["role0", "content0"]).alias("first"),
        pl.struct(["role1", "content1"]).alias("second"),
        pl.struct(["role2", "content2"]).alias("third")) \
        .select(
        pl.col("zero"),
        pl.col("first").struct.rename_fields(["role", "content"]),
        pl.col("second").struct.rename_fields(["role", "content"]),
        pl.col("third").struct.rename_fields(["role", "content"])
    ) \
        .select(
        pl.col("zero").arr.concat(
            ["first", "second", "third"]).alias("messages")
    ) \
        .select(
        pl.lit("gpt-3.5-turbo").alias("model"),
        pl.col("messages")
    ) \
        .collect()


def parse_transribed(file_path):
    try:
        transcribed = pl.read_csv(file_path,
                                  has_header=False,
                                  new_columns=['group', 'res']).lazy() \
            .sort(by='group')
    except:
        transcribed = pl.DataFrame(pd.read_csv(file_path,
                                               names=['group', 'res'])).lazy().sort(by='group')

    transcribed = transcribed \
        .with_columns(pl.col("res").str.split(by="。", inclusive=True).alias("split_sentence")) \
        .with_columns(pl.col("split_sentence").arr.lengths().alias("num_sentences")).collect()

    return transcribed


def get_each_batch(trans: pl.DataFrame, to_pandas: bool = False) -> Dict[int, pl.DataFrame]:
    trans = trans.lazy().explode("split_sentence").with_columns(pl.col("split_sentence").str.split(".")) \
        .select(
            pl.col("group"),
            pl.col("split_sentence").arr.get(0).str.extract(
                r"(\d+)").cast(pl.Int64).alias("line"),
            pl.col("split_sentence").arr.get(1).alias("transcribed")
    ).collect()
    if to_pandas:
        return {idx: val for (idx, val) in trans.to_pandas().groupby("group")}
    return trans.partition_by("group", maintain_order=True, as_dict=True)


def align_ori_trans(ori: pl.DataFrame, trans: pl.DataFrame) -> pl.DataFrame:
    combined = ori.join(
        trans.lazy().explode("split_sentence").with_columns(
            pl.col("split_sentence").str.split("."))
        .select(
            pl.concat_str([pl.col("group"),
                           pl.col("split_sentence").arr.get(0).str.extract(r".*(\d)")]).cast(pl.Int64).alias("line"),
            pl.col("split_sentence").arr.get(1).alias("transcribed"))
        .with_columns(
            pl.when(pl.col("line") % 10 == 0).then(
                pl.col("line") + 10).otherwise(pl.col("line"))
        ).unique(subset=['line']).collect(),
        left_on="is_end",
        right_on="line",
        how="left"
    )
    print(combined.select(pl.col("transcribed").null_count()))
    return combined.with_columns(
        pl.col("transcribed").fill_null(strategy="forward"),
        pl.col("is_end").cast(pl.Utf8))


def check_consistency(ori: pl.DataFrame, res: pl.DataFrame, group: int):
    pass
