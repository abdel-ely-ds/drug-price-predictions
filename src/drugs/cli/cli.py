import logging
import os.path

import click
import pandas as pd

from drugs.core.trainer import Trainer
from drugs.utils import get_latest_run_id


@click.command()
@click.option("data-dir", type=str, required=True)
@click.option("raw-df-name", type=str, required=True)
@click.option("ingredients-df-name", type=str, required=True)
@click.option("--infer-mode", is_flag=True)
@click.option("--output-dir", type=str, required=False)
@click.option("--run-id", type=int, required=False)
def run(
    data_dir: str,
    raw_df_name: str,
    ingredients_df_name: str,
    infer_mode: bool = False,
    output_dir: str = None,
    run_id: int = get_latest_run_id(),
) -> None:
    msg = "infer mode" if infer_mode else "training mode"
    click.echo(f"running on a {msg}")
    click.echo(f"using run id: {run_id}")

    trainer = Trainer()
    raw_df = pd.read_csv(os.path.join(data_dir, raw_df_name))
    ingredients_df = pd.read_csv(os.path.join(data_dir, ingredients_df_name))

    if infer_mode:
        trainer.load_artifacts(
            from_dir=output_dir,
            run_id=run_id,
        )
        predictions = trainer.predict(raw_df=raw_df, ingredient_df=ingredients_df)

        trainer.save_predictions(predictions=predictions, output_dir=output_dir)

    else:
        trainer.train(raw_df=raw_df, ingredient_df=ingredients_df)
        trainer.save_artifacts(output_dir=output_dir)


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(run)
    command_group()
