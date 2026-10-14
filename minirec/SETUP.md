## Setup guide

The steps below describe what you need to do to get new tasks, solve them and test them.

You will go through the following steps:
- [Pre-setup](#pre)
- [Setting up the git repository](#git)
- [Installing Python and packages](#install)
- [Solving the tasks](#send)

---

### Pre-setup <a id='pre'/>

<details><summary><a>MacOS</a></summary>

MacOS lacks a package manager for convenient installation of things (like apt-get).

So you need to install `brew` following the [official instructions](https://brew.sh/).
```shell
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
</details>

<details><summary><a>Windows</a></summary>

Windows 10 introduced a feature called WSL: Windows Subsystem for Linux,
which lets you run Linux applications on Windows.
We recommend using it and following the instructions below
as if you had Linux as your operating system.

Install `wsl` following the [official instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

When choosing a Linux distribution, take Ubuntu 22.04 or 24.04.

Start the installed system. On login you will be in the `/home/<username>` directory;
to be able to work with the code from Windows itself (for example, in PyCharm),
we recommend placing the task directory at `"/mnt/c/Users/<username>/My Documents"`,
which is available in Windows as `C:\Users\<username>\My Documents`.

Go to that directory:
```bash
$ cd "/mnt/c/Users/<username>/My Documents"
```
Continue with the instructions below

</details>

### Setting up the git repository <a id='git'/>

All work on getting and solving the tasks goes through a git repository. First you need to clone it.
What git is and how to work with it was covered in the [lecture about git](https://yadi.sk/i/YUe3SJYo11EChA).

#### Installing the prerequisites

We need to install git.
There is a good chance it is already installed; you can check like this:
```shell
$ git --version
```
If not, install it
<details><summary><a>Linux</a></summary>

On Ubuntu/Debian it is simple:

```bash
$ sudo apt-get install git
```

If you have another distribution, you probably know how to install a package there.
</details>

<details><summary><a>MacOS</a></summary>

On MacOS `git` is already installed if you installed `xcode`

If you want to install git separately, install it via brew:
```bash
$ brew install git
```

</details>

#### Cloning and setting up the repository

```bash
# Go to the directory where you want to keep the repository with the tasks,
# usually the home directory - `/home/<username>` or /Users/<username>, also written as `~` (a tilde, by the way)
$ cd <your chosen directory>

# Clone the repository
$ git clone https://github.com/yandexdataschool/recsys_course.git

# Go to the directory with the tasks
$ cd recsys_course/minirec

# Configure git so that it knows who you are
$ git config --local user.name "<your name>"
$ git config --local user.email "<your email>"
```

### Build, tests <a id='install'/>

We use Python version `3.13.5`.
NB: If you have a different Python version, we cannot guarantee that all tasks and tests work correctly.

You also need to install [uv](https://github.com/astral-sh/uv) following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/#installing-uv). It is used to manage environments, the builds of the different parts of the framework and the dependencies via the `pyproject.toml` files. Running the tests differs from task to task, so look for the exact commands for local testing in the task descriptions.

The repository root also has a `pyproject.toml`: it is needed for running the tests, and you can use it to create a virtual environment and index the code in your IDE.

- Create a virtual environment in the repository with the tasks
  ```bash
  $ cd <path to the cloned repository>/minirec
  $ uv venv --python 3.13.5 rec_env
  ```

- Activate the virtual environment (it stays active until you close the console or run `deactivate`)
  ```bash
  $ source rec_env/bin/activate
  ```  
  The following prefix appears:
  ```bash
  (rec_env)$ ...
  ```

- Check that everything is activated and `python` points to the `rec_env` folder
  ```bash
  $ which python
  <directory with the tasks>/rec_env/bin/python
  ```

- Install the dependencies with uv from `pyproject.toml`:
  ```bash
  # install the dependencies from pyproject.toml
  $ uv sync --active
  ```

### Solving the tasks <a id='send'/>


#### Getting new tasks
To get new tasks, run `git pull`: it downloads the latest updates of the repository.

#### Solving a task
Each task has its own week and task directory (`tasks/basics_and_services/experiments_server` and so on); what matters to us is its contents:
- the statement and the instructions for testing it locally are in the `README.md` file
- the task code lives in different parts of our framework, as do the public tests for the task

Check your solution with the commands from the task's `README.md`, for example:
```bash
(cd experiments_server && uv run pytest)
```
