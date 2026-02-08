[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "asa"
version = "0.1.0"
description = "A package for Addressed State Attention"
authors = ["Justin Brown <digitaldaimyo@gmail.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.8"
torch = "^2.0"

[tool.poetry.dev-dependencies]
pytest = "^6.0"
