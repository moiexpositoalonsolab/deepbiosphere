### Development Files

1. `dev.Dockerfile` provides the Docker build environment, which pulls an image of a CUDA-enabled Ubuntu image, installs some development dependencies via `apt`, installs `conda`, and then installs python and its necessary dependencies.

2. `environment.yml` which configures the conda python requirements, channels to install from, and necessary dependencies.

3. `devcontainer.json` which configures where to build the docker container (as of now, `../`, which builds in the root of the repo), mounts the `.ssh` directory to allow git access, and enables some VSCode extensions within the repo. **Note that since the docker container is an isolated environment, you will not be able to use your local extensions inside the container within specifying them here.**

#### VSCode Dev Container

First off, ensure Docker is installed, VSCode is installed, and the `Dev Containers` extension is installed. Then, with the repo open in VSCode, run the following command using the command pallete:

`Cmd + Shift + P > "Dev Containers: Reopen in Dev Container`

This will build the dev container, run it, and connect to it via ssh.

#### DevPod Dev Container Call

DevPod takes the same `devcontainer.json` and does some extra work to prepare your development environment - as of now, this call performs the same as the above:

```
devpod up git@github.com:SchmidtDSE/deepbiosphere@main --devcontainer-path .devcontainer/devcontainer.json --debug  --ide vscode
```

Other flags / considerations:

- The @`branch` format within the git path - this allows you to clone a particular branch for development.

- The flag `--recreate` will re-run any relevant bits in `devcontainer.json` and `dev.Dockerfile`.

- The **more extreme** flag `--reset` will re-clone the repo and re-run any relevant bits in `devcontainer.json` and `dev.Dockerfile`. **This will wipe any local changes!**

It is recommended to `--recreate` and make sure everything still works before merging any PRs.

_DevPod also provides `providers` to deploy the very same dev environment to cloud infrastructure (`AWS`, `GCP`, `Azure`, etc) to more seamlessly scale up and deploy. Information will be provided here when this is implemented._

#### Important Notes

- pip and conda don't play nice - for now, you will need to excute the following after build:

  - conda init
  - restart bash or start a new terminal instance
  - `conda activate deepbiosphere`
  - `pip install -e .` to use `setup.py` to install dependencies

- In order to connect to GBIF, you will need to configure `~/.netrc` with your GBIF credentials. The Docker build script will create this file with dummy credentials, but you will need to fill in your _real_ credentials.

- You will also need a .env in the _root of the repo_ that contains the same contents as `example.env`. Some of this you shouldn't have to change (mostly, the paths to relevant files inside the running container), but some is specific to you. Note that this needs to be done in addition to the above `~/.netrc` stuff.
