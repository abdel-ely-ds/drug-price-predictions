from click.testing import CliRunner

from drugs.cli import cli


class TestCli:
    def test_run_train(self):
        args = [
            "--train",
            "--data-dir",
            "../data/ ",
            "--output-dir",
            "./artifacts ",
            "--df-filename",
            "drugs_train.csv ",
            "--df-ingredient-filename",
            "active_ingredients.csv ",
        ]

        runner = CliRunner()
        result = runner.invoke(cli.train, args=args)
        expected_results = "running on training mode\n"
        assert result.output == expected_results

    def test_run_train_predict(self):
        args = [
            "--train",
            "--predict",
            "--data-dir",
            "../data/",
            "--output-dir",
            "./artifacts",
            "--df-filename",
            "drugs_train.csv",
            "--df-ingredient-filename",
            "active_ingredients.csv",
        ]

        runner = CliRunner()
        result = runner.invoke(cli.train, args=args)
        assert result.exit_code != 0
