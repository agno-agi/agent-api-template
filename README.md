## App Workspace Template

This repo contains the code for running an agent-app and supports 2 environments:

1. **dev**: A development environment running locally on docker
2. **prd**: A production environment running on AWS ECS

## Setup Workspace

1. [Install uv](https://docs.astral.sh/uv/#getting-started): `curl -LsSf https://astral.sh/uv/install.sh | sh`

> from the `api-ws-template` dir:

2. Install workspace and activate the virtual env:

```sh
./scripts/install.sh
source .venv/bin/activate
```

3. Copy `workspace/example_secrets` to `workspace/secrets`:

```sh
cp -r workspace/example_secrets workspace/secrets
```

4. Optional: Create `.env` file:

```sh
cp example.env .env
```

## Run Api Workspace Template locally

1. Install [docker desktop](https://www.docker.com/products/docker-desktop)

2. Set OpenAI Key

Set the `OPENAI_API_KEY` environment variable using

```sh
export OPENAI_API_KEY=sk-***
```

**OR** set in the `.env` file

3. Start the workspace using:

```sh
ag ws up dev
```

Open [localhost:8000/docs](http://localhost:8000/docs) to view the FastAPI docs.
Open [localhost:8501](http://localhost:8501) to view the Streamlit App.

4. Stop the workspace using:

```sh
ag ws down dev
```

## Next Steps:

- Run the App Workspace Template on AWS - Docs coming soon
