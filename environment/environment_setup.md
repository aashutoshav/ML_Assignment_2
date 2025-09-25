# Environment Setup

1. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Use the command: `conda env create -f ml_hw2_env_<OS_ARCH>.yml`
    - Make sure to modify the command based on your architecture (`linux_64`, `osx_64`, `osx_arm64`, or `win_64`).
    - If you're not on one of these architectures, try the closest match.
    - We can't explicitly guarantee that the environment will solve correctly, but we generally try to use packages that are not platform dependent.
    - This should create an environment named `ml-hw2`.
2. Activate the conda environment:
    - `conda activate ml-hw2`

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
