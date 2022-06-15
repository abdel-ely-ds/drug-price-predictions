import nox
from nox.sessions import Session

locations = "src", "tests", "noxfile.py"
nox.options.sessions = "tests", "blacken", "build"
nox.options.stop_on_first_error = True


@nox.session(python=["3.7", "3.8"], reuse_venv=True)
def blacken(session: Session) -> None:
    """
    Run black code formatter
    """
    args = session.posargs or locations
    session.install("black==22.3.0", "isort==5.6.4")
    session.run("isort", *args)
    session.run("black", *args)


@nox.session(python=["3.7", "3.8"], reuse_venv=True)
def tests(session: Session) -> None:
    """
    Test code
    """
    session.install("pip", "install", ".[test]")
    session.run("pytest")


@nox.session(python=["3.7", "3.8"], reuse_venv=True)
def build(session: Session) -> None:
    args = session.posargs or ["--format=zip"]
    session.run("python3", "setup.py", "sdist", *args)
