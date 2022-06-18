import logging
import os

import click
import pandas as pd

from drugs.drugs import Drugs
from drugs.exceptions.exceptions import MultipleModes, NoModeSpecified


@click.command()
@click.option("--train", is_flag=True)
@click.option("--predict", is_flag=True)
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
@click.option("--df-ingredient-filename", type=str, required=True)
@click.option("--val-df-filename", type=str, required=False)
@click.option("--val-df-ingredient-filename", type=str, required=False)
@click.option("--from-dir", type=str, required=False)
@click.option("--early-stopping-rounds", type=int, required=False)
@click.option("--verbose", type=int, required=False)
def run(
    train: bool,
    predict: bool,
    data_dir: str,
    output_dir: str,
    df_filename: str,
    df_ingredient_filename: str,
    val_df_filename: str = None,
    val_df_ingredient_filename=None,
    from_dir: str = None,
    early_stopping_rounds: int = 20,
    verbose: bool = True,
) -> None:
    if not predict and not train:
        raise NoModeSpecified()

    if predict and train:
        raise MultipleModes()

    msg = "training mode" if train else "inference mode"
    click.echo(f"running on {msg}")

    drugs = Drugs()
    df = pd.read_csv(os.path.join(data_dir, df_filename))
    df_ingredient = pd.read_csv(os.path.join(data_dir, df_ingredient_filename))

    try:
        val_df = pd.read_csv(os.path.join(data_dir, val_df_filename))
        val_df_ingredient = pd.read_csv(
            os.path.join(data_dir, val_df_ingredient_filename)
        )

    except FileNotFoundError:
        val_df, val_df_ingredient = None, None

    if predict:
        drugs.load_artifacts(
            from_dir=from_dir,
        )
        predictions = drugs.predict(df=df, df_ingredient=df_ingredient)

        drugs.save_predictions(predictions=predictions, output_dir=output_dir)

    if train:
        drugs.fit(
            df=df,
            df_ingredient=df_ingredient,
            val_df=val_df,
            val_df_ingredient=val_df_ingredient,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )

        drugs.save_artifacts(output_dir=output_dir)


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(run)
    command_group()
