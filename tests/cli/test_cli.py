from click.testing import CliRunner

from drugs.cli import cli


class TestCli:
    def test_run(self):
        args = [
            "--train",
            "--date-dir" "../data/",
            "--output-dir",
            "./artifacts",
            "--df-filename",
            "drugs_train.csv",
            "--df-ingredient-filename",
            "active_ingredients.csv",
        ]

        runner = CliRunner()
        result = runner.invoke(cli.run, args=args)
        expected_input_file_path = "./data/fake_users.csv"
        expected_output_dir = "./artifact"
        expected_val_mode = True
        assert (
            result.output
            == f"input path of data: {expected_input_file_path}\noutput path of artifacts: {expected_output_dir}\n"
            f"training mode: {expected_val_mode}\n"
        )
