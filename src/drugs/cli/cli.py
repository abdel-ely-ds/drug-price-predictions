import logging

import click

from drugs.cli.predict import predict
from drugs.cli.train import train


@click.group()
def command_group():
    pass


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger(__package__).setLevel(logging.INFO)
    command_group.add_command(train)
    command_group.add_command(predict)
    command_group()
