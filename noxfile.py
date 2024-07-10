import nox


@nox.session
def tests(session):  # type: ignore
    session.install(".[tests]")
    session.install("typing-extensions")
    session.run("pytest", "--cov=mixscatter", "tests/")


@nox.session
def lint(session):  # type: ignore
    session.install(".[lint]")
    session.run("ruff", "format")
    session.run("ruff", "check", "src", "tests", "--fix", "--line-length=120")


@nox.session
def type_check(session):  # type: ignore
    session.install(".[type-check]")
    session.run("mypy", "src")


@nox.session
def docs(session):  # type: ignore
    session.install(".[docs]")
    session.run("mkdocs", "build")
