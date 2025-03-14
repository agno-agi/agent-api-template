## Sample Agentic Application

This repo contains the code for a running an agentic application with:

1. A FastAPI server
2. A Postgres database with the PgVector extension.

You can run the agentic application in 2 environments:

1. A development environment running locally on docker
2. A production environment running on AWS ECS

## Setup

1. [Install uv](https://docs.astral.sh/uv/#getting-started): `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. Create a virtual environment and install dependencies:

```sh
./scripts/dev_setup.sh
```

3. Activate virtual environment

```
source .venv/bin/activate
```

## Run application locally using docker

1. Install [docker desktop](https://www.docker.com/products/docker-desktop)

2. Export API keys

Required: Set the `OPENAI_API_KEY` environment variable using

```sh
export OPENAI_API_KEY=***
```

> you may use any model provider, just need to update the /agents

Optional: Set the `EXA_API_KEY` if you'd like to use Exa search

```sh
export EXA_API_KEY=***
```

3. Start the workspace:

```sh
ag ws up
```

- This will run 3 containers:
  - FastAPI on [localhost:8000](http://localhost:8000/docs)
  - Postgres on [localhost:5432](http://localhost:5432)
- Open [localhost:8501](http://localhost:8501) to view the Streamlit App.
- Open [localhost:8000/docs](http://localhost:8000/docs) to view the FastAPI docs.

4. Stop the workspace using:

```sh
ag ws down
```

## Learn more:

- Learn more about this template and how to customize it.
