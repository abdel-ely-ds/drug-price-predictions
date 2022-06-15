import logging

import click
import pandas as pd

from drugs.core.trainer import Trainer


@click.command()
@click.option("--input-file-path", type=str, required=True)
@click.option("--random-state", type=int, required=False, default=42)
@click.option("--col-names", type=list, required=False)
@click.option("--label-name", type=str, required=False)
@click.option("--output-dir", type=str, required=True)
@click.option("--val-mode", is_flag=True)
def run(
    input_file_path: str,
    output_dir: str,
    random_state: int = 42,
    col_names: str = None,
    label_name: str = None,
    val_mode: bool = False,
) -> None:
    click.echo(f"input path of data: {input_file_path}")
    click.echo(f"output path of artifacts: {output_dir}")
    click.echo(f"training mode: {val_mode}")
    spotter = Spotter(
        random_state=random_state,
        col_names=col_names,
        label_name=label_name,
        output_dir=output_dir,
        val_mode=val_mode,
    )
    raw_data = pd.read_csv(input_file_path)
    if val_mode:
        spotter.predict(raw_test=raw_data)

    else:
        spotter.run(raw_data=raw_data)


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(run)
    command_group()
