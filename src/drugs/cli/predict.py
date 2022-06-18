import os

import click
import pandas as pd

from drugs.drugs import Drugs


@click.command()
@click.option("--data-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--from-dir", type=str, required=True)
@click.option("--df-filename", type=str, required=True)
@click.option("--df-ingredient-filename", type=str, required=True)
def predict(
    data_dir: str,
    output_dir: str,
    from_dir: str,
    df_filename: str,
    df_ingredient_filename: str,
) -> None:
    click.echo(f"Inference started...")

    drugs = Drugs()
    df = pd.read_csv(os.path.join(data_dir, df_filename))
    df_ingredient = pd.read_csv(os.path.join(data_dir, df_ingredient_filename))

    drugs.load_artifacts(
        from_dir=from_dir,
    )
    predictions = drugs.predict(df=df, df_ingredient=df_ingredient)

    drugs.save_predictions(predictions=predictions, output_dir=output_dir)
