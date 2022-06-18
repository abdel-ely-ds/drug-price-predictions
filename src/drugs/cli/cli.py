import logging
import os

import click
import pandas as pd

from drugs.drugs import Drugs
from drugs.utils.utils import (
    MultipleModes,
    NoModeSpecified,
    get_latest_run_id,
    merge_dfs,
)


@click.command()
@click.option("--train", is_flag=True)
@click.option("--predict", is_flag=True)
@click.option("--data-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
@click.option("--df-ingredient-filename", type=str, required=True)
@click.option("--output-dir", type=str, required=False)
@click.option("--from-dir", type=str, required=False)
@click.option("--run-id", type=int, required=False)
def run(
    train: bool,
    predict: bool,
    data_dir: str,
    df_filename: str,
    df_ingredient_filename: str,
    val_df_filename: str = None,
    val_df_ingredient_filename=None,
    verbose: bool = True,
    early_stopping_round: int = 20,
    output_dir: str = None,
    from_dir: str = None,
    run_id: int = get_latest_run_id(),
) -> None:

    if not predict and not train:
        raise NoModeSpecified()

    if predict and train:
        raise MultipleModes()

    msg = "training mode" if train else "inference mode"
    click.echo(f"running on {msg}")
    click.echo(f"using run id: {run_id}")

    drugs = Drugs()
    df = pd.read_csv(os.path.join(data_dir, df_filename))
    df_ingredient = pd.read_csv(os.path.join(data_dir, df_ingredient_filename))

    if predict:
        drugs.load_artifacts(
            from_dir=from_dir,
            run_id=run_id,
        )
        predictions = drugs.predict(df=df, df_ingredient=df_ingredient)

        drugs.save_predictions(predictions=predictions, output_dir=output_dir)

    if train:
        drugs.fit(df=df, df_ingredient=df_ingredient)
        drugs.save_artifacts(output_dir=output_dir)


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(run)
    command_group()
