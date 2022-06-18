import os

import click
import pandas as pd

from drugs.drugs import Drugs


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
@click.option("--df-ingredient-filename", type=str, required=True)
@click.option("--val-df-filename", type=str, required=False)
@click.option("--val-df-ingredient-filename", type=str, required=False)
@click.option("--early-stopping-rounds", type=int, required=False)
@click.option("--verbose", type=int, required=False)
def train(
    data_dir: str,
    output_dir: str,
    df_filename: str,
    df_ingredient_filename: str,
    val_df_filename: str = None,
    val_df_ingredient_filename=None,
    early_stopping_rounds: int = 20,
    verbose: bool = True,
) -> None:
    click.echo(f"Training started...")

    drugs = Drugs()
    df = pd.read_csv(os.path.join(data_dir, df_filename))
    df_ingredient = pd.read_csv(os.path.join(data_dir, df_ingredient_filename))

    try:
        val_df = pd.read_csv(os.path.join(data_dir, val_df_filename))
        val_df_ingredient = pd.read_csv(
            os.path.join(data_dir, val_df_ingredient_filename)
        )
    except TypeError:
        val_df, val_df_ingredient = None, None

    drugs.fit(
        df=df,
        df_ingredient=df_ingredient,
        val_df=val_df,
        val_df_ingredient=val_df_ingredient,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds,
    )

    drugs.save_artifacts(output_dir=output_dir)
