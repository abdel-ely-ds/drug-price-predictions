import logging
import os.path

import click
import pandas as pd

from drugs.core.trainer import Trainer
from drugs.utils import MultipleModes, NoModeSpecified, get_latest_run_id, merge_dfs


@click.command()
@click.option("data-dir", type=str, required=True)
@click.option("raw-df-name", type=str, required=True)
@click.option("ingredients-df-name", type=str, required=True)
@click.option("--train", is_flag=True)
@click.option("--predict", is_flag=True)
@click.option("--output-dir", type=str, required=False)
@click.option("--from-dir", type=str, required=False)
@click.option("--run-id", type=int, required=False)
def run(
    data_dir: str,
    raw_df_name: str,
    ingredients_df_name: str,
    train: bool = False,
    predict: bool = False,
    output_dir: str = None,
    from_dir: str = None,
    run_id: int = get_latest_run_id(),
) -> None:

    if not predict and not train:
        raise NoModeSpecified()

    if predict and train:
        raise MultipleModes()

    msg = "training mode" if train else "inference mode"
    click.echo(f"running on a {msg}")
    click.echo(f"using run id: {run_id}")

    trainer = Trainer()
    raw_df = pd.read_csv(os.path.join(data_dir, raw_df_name))
    ingredient_df = pd.read_csv(os.path.join(data_dir, ingredients_df_name))
    df = merge_dfs(raw_df, ingredient_df)
    
    if predict:
        trainer.load_artifacts(
            from_dir=from_dir,
            run_id=run_id,
        )
        predictions = trainer.predict(df=df)

        trainer.save_predictions(predictions=predictions, output_dir=output_dir)

    if train:
        trainer.train(df=df)
        trainer.save_artifacts(output_dir=output_dir)


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(run)
    command_group()
