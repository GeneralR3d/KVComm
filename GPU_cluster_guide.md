# CCDS GPU Cluster (TC1) — Detailed User Guide

**Classification: [RESTRICTED]**  
**Source:** `GPU_cluster_guide.pdf`, 31 pages, titled *CCDS GPU Cluster (TC1) - USER GUIDE*.  
**Source last updated:** 12 February 2026.

This is a consolidated Markdown conversion of the source guide, including the operational information shown in its screenshots. It preserves the setup instructions, commands, examples, reference links, usage rules, tips, warnings, resource application procedures, and support details. The original section order and page references are retained below. Repeated page headers and footers are consolidated here; screenshots are described or transcribed into searchable text.

The cluster inventory, software versions, account limits, IP addresses in examples, and policies below are those recorded in the source. This conversion does not establish their current live values. In particular, **check your own assigned QoS and the available partition before submitting jobs**. Source inconsistencies and small transcription corrections are documented in [Conversion notes and source inconsistencies](#conversion-notes-and-source-inconsistencies).

Commands do not include the source's `>>` prompt markers. Replace placeholders such as `<username>`, `<jobid>`, `<version>`, and `XXXX` before use; do not type the angle brackets literally. A block containing alternative commands is a reference, not necessarily a sequence to run in full.

## Contents

- [Essential rules: what to do and what not to do](#essential-rules-what-to-do-and-what-not-to-do)
- [1. Introduction](#1-introduction) — original pages 2–3
- [2. Logging into the cluster, transferring files, and maintaining storage](#2-logging-into-the-cluster-transferring-files-and-maintaining-storage) — original pages 4–6
- [3. Conda packages, environments, and CUDA](#3-conda-packages-environments-and-cuda) — original pages 7–13
- [4. SLURM user guide](#4-slurm-user-guide) — original pages 14–15
- [5. Guidelines for job submission](#5-guidelines-for-job-submission) — original pages 16–22
- [6. Shared resources](#6-shared-resources) — original page 23
- [7. Miscellaneous: Jupyter, SSH tunnelling, and monitoring](#7-miscellaneous-jupyter-ssh-tunnelling-and-monitoring) — original pages 24–29
- [8. Applying for additional resources](#8-applying-for-additional-resources) — original page 30
- [9. Important notices and support](#9-important-notices-and-support) — original page 31
- [Conversion notes and source inconsistencies](#conversion-notes-and-source-inconsistencies)

## Essential rules: what to do and what not to do

### Do

- Use the cluster **only for the academic purpose for which your access was granted**.
- Connect from the NTU network; connect to the NTU VPN first when off campus.
- Make your first connection using SSH. **Your home directory is created only after your first login.**
- Use your **NTU Network Account ID in lowercase** and its password.
- Keep your work within your assigned home directory and authorised shared folders.
- Use the head node for the documented access, file transfer, environment preparation, script editing, submission, and job-management steps.
- Run computation on compute nodes through a SLURM job script, normally submitted with `sbatch`.
- Verify your actual QoS using `MyTCinfo`; request only an assigned QoS and keep resource requests within its limits, including totals across jobs.
- Create your own Conda environment for installing packages. Prefer `conda install`; use `pip install` only when the required package is unavailable through Conda, as advised in the source.
- Verify GPU/CUDA operation inside a compute-node allocation, for example through a Jupyter terminal launched by a SLURM job.
- Inspect transferred scripts for invalid characters and incorrect file formats before submitting them.
- Regularly check disk usage, remove unwanted data and environments, and back up results to your own device.
- Review completed-job CPU and memory efficiency with `seff`, and GPU usage through the provided logs, to improve later resource requests.
- Maintain the SSH tunnel while using Jupyter. Shut down Jupyter or cancel its job when finished so CPU, memory, and GPU resources return to the pool.

### Do not

- **Do not compile or run your computation on the head node `CCDS-TC1`.** High CPU/memory processes there can be terminated without notice; repeat offenders can be banned.
- **Do not run GPU/CUDA hardware checks such as `nvidia-smi`, `nvcc --version`, or `nvcc -V` on the head node.** The GPUs are in the compute nodes.
- Do not use an unassigned QoS. Such jobs can be terminated without prior notice.
- Do not exceed your account's CPU, memory, GPU, job-count, node-count, or wall-time limits. Two submitted jobs do not receive two independent copies of a per-user resource allowance.
- Do not install packages into the shared Conda **base** environment. Users do not have permission to use `sudo` or install system applications.
- Do not casually delete dotfiles or system configuration directories such as `.bash_profile`, `.bashrc`, and `.config` while cleaning storage.
- Do not access other users' directories or other unauthorised files, even just to browse.
- Avoid `srun` for the job-submission workflow described in this guide, especially for computations longer than an hour. Use `sbatch` and reconnect later for results.
- Do not use Jupyter as the normal launcher for finished computation. Use it for development, analysis, verification, and debugging; submit ready code directly as a job.
- Do not leave an unused Jupyter job holding resources.
- Do not assume the cluster backs up your data. **There is no data backup service.**

## 1. Introduction

*Original pages 2–3.*

### 1.1 Cluster structure and responsibilities

TC1 has two types of nodes:

- **Head node:** the node you initially log into and the main access point to the cluster.
- **Compute nodes:** the nodes containing GPU cards and running allocated computations.

The source's architecture diagram shows the following access path and components:

```text
Off-campus Internet
        |
     NTU VPN
        |
    NTU Network
        |
      Firewall
        |
  TC1 GPU Cluster
        +-- TC1 Head Node
        +-- Compute Nodes 1, 2, 3, 4, 5, 6, 7
        +-- Cluster Storage Node
```

The head node, compute nodes, and storage node are connected within the cluster. **The GPUs are in the compute nodes, not the head node.**

Users must create a job script and submit their computation request, described in the source as a **non-interactive job**, to **SLURM — Simple Linux Utility for Resource Management**. The scheduler processes the job on allocated compute nodes. Users are not allowed to compile or run their code on the head node.

The resources available to a user are limited by the assigned **QoS — Quality of Service**.

### 1.2 Client tools

#### PuTTY: SSH client for Windows

PuTTY is open-source SSH and Telnet client software for Windows. The guide describes SSH as a protocol providing secure access to a computer over an unsecured network.

**Tip:** the portable `putty.exe` edition does not need installation and is ready to use.

Source download and reference links:

- [PuTTY download page](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html)
- [SSH.com PuTTY guide](https://www.ssh.com/academy/ssh/putty)

#### SSH access for macOS

Use SSH from Terminal, or install another SSH client to facilitate the connection. The source provides:

- [PuTTY and SSH on Mac](https://www.ssh.com/academy/ssh/putty/mac)
- [PuTTY download information](https://www.puttygen.com/download-putty)
- [Termius SSH client for macOS](https://termius.com/free-ssh-client-for-mac-os)

#### WinSCP: SFTP client for Windows

WinSCP, Windows Secure Copy, is a free, open-source SFTP, FTP, WebDAV, and SCP client for Microsoft Windows. **SFTP — SSH File Transfer Protocol** — provides encrypted/secure file transfer.

**Installation tip:** select **Explorer** as the user interface style.

- [Official WinSCP site](https://winscp.net/eng/index.php)
- [WinSCP installation guide](https://winscp.net/eng/docs/guide_install)
- [Additional WinSCP guide](https://www.puttygen.com/winscp)

#### FileZilla Client: SFTP client for macOS

- [FileZilla download site](https://filezilla-project.org/index.php)
- [Connecting to SFTP on macOS](https://lemp.io/how-to-connect-to-sftp-mac-os/)
- [Using Mac Terminal as an FTP/SFTP client](https://beebom.com/how-to-use-mac-terminal-ftp-sftp-client/)
- [Additional Mac Terminal FTP/SFTP guide](https://tipsmake.com/use-terminal-on-mac-as-ftp-or-sftp-client)

### 1.3 Assumed knowledge and further reading

The guide provides general information to get started in TC1. It assumes you already know how to:

- Connect to a Linux server using SSH.
- Transfer files between a local computer and a Linux server.
- Execute Linux commands at a command prompt.

For more detailed information, use the source's suggested references or search for the information you need.

Basic Linux commands:

- [Serverdale Linux commands](https://serverdale.com/en/linux-commands)
- [CentOSHelp Linux system commands](https://centoshelp.org/resources/commands/linux-system-commands/)

Redirecting or piping program output into a file:

- [Introduction to Linux: I/O redirection](https://tldp.org/LDP/intro-linux/html/sect_05_01.html)
- [Piping output to a file](https://linuxconfig.org/how-to-pipe-output-to-a-file-on-linux)
- [Saving Python terminal output to a text file](https://www.geeksforgeeks.org/ways-to-save-python-terminal-output-to-a-text-file/)

### 1.4 End-to-end workflow

1. **Establish your first SSH connection.** Your home directory is created after this first login.
2. Upload the required job scripts, code, and datasets from your local machine to your home directory using an SFTP client.
3. Set up and configure your Conda environment with the applications required for computation.
4. Verify that environment **on a compute node**, using Jupyter as described in the CUDA and Jupyter sections. The original cross-reference is page 11 onwards.
5. Remember that GPUs are absent from the head node. Do not execute your code or hardware checks such as `nvidia-smi` and `nvcc --version` there.
6. Submit the non-interactive job through SLURM so your code runs on the assigned compute node. The original job-submission guidance starts on page 16.
7. Copy the job output back to your local machine using SFTP.

## 2. Logging into the cluster, transferring files, and maintaining storage

*Original pages 4–6.*

### 2.1 Network access and credentials

The CCDS GPU cluster is accessible **only within the NTU network**. For off-campus access, connect through the **NTU VPN** first.

- **Head-node hostname:** `CCDS-TC1`
- **Head-node IP address:** `10.96.189.11`
- **SSH/SFTP port:** `22`
- **Username:** your NTU Network Account ID, **in lowercase**
- **Password:** your NTU Network Account password

VPN resources from the source:

- [NTU VPN portal](https://vpngate.ntu.edu.sg)
- [GlobalProtect VPN instructions](https://www3.ntu.edu.sg/cits2/ras/for_cms_ref/GlobalProtectVPN.pdf)
- [GlobalProtect software page](https://vpngate.ntu.edu.sg/global-protect/getsoftwarepage.esp)

### 2.2 SSH with PuTTY

1. Launch PuTTY.
2. Set **Host Name / IP address** to `10.96.189.11`.
3. Set **Port** to `22` and use **SSH** as the connection type.
4. Open the connection.
5. On the first connection, PuTTY asks you to accept the host key. The source instructs you to click **Yes** to continue.
6. Enter your lowercase NTU Network Account ID and password when prompted.

The screenshot's login notice repeats the academic-use restriction, monitoring policy, no-backup policy, prohibition on unauthorised browsing, availability of the guide in the shared area, `MyTCinfo`, and support contact. These notices are preserved in the relevant sections below.

### 2.3 SSH from Linux or macOS Terminal

Either of these source examples establishes the connection:

```bash
ssh -l <username> 10.96.189.11
ssh -p 22 <username>@10.96.189.11
```

Replace `<username>` with your lowercase NTU account ID.

**Best practice when leaving an SSH session:** type `exit` or press **Ctrl+D**.

### 2.4 File transfer with WinSCP or FileZilla

The source demonstrates WinSCP, with the same SFTP connection details applicable to FileZilla:

1. Launch the SFTP client.
2. Set **File protocol** to **SFTP**.
3. Set **Host name** to `10.96.189.11`.
4. Set **Port number** to `22`.
5. Enter your lowercase NTU Network Account ID and password.
6. Confirm that the remote pane is showing your correct home directory before transferring files.

To check your home-directory path from the SSH session:

```bash
pwd
```

The example account `scetest5` has the sample path `/home/UG/scetest5`. Use the path returned for **your own account**.

- To **upload**, select a file on your local machine and drag it into the remote window. The Upload window transfers it to your cluster home directory.
- To **download**, transfer a file from your home directory to the selected local disk/folder on your machine. The Download window handles this direction.

### 2.5 Home-directory limits and risks

Your data and computation are bounded by your assigned home directory and shared folder, if any.

All user directories have disk quotas. The source says users may receive **at least 100 GB**, with the actual assignment depending on the account and usage. Check your own allocation rather than assuming a fixed entitlement.

When quota usage exceeds **98%**, you may experience:

- Inability to log in.
- Inability to compute/process data.
- Inability to save new data, or loss of data.

Regular housekeeping is necessary to keep sufficient free space for computation.

### 2.6 Inspecting disk usage

Run the following in your home directory:

```bash
ncdu
```

The screenshot shows `ncdu` scanning `/home/UG/scetest5` and listing hidden directories and files such as `.ssh`, `.mozilla`, `.bash_history`, `.bashrc`, `.emacs`, `.kshrc`, `.bash_profile`, `.k5login`, `.lesshst`, `.bash_logout`, and `.python_history`. Its interface says to use the arrow keys to navigate and `?` for help.

Press **q** to abort scanning or exit.

Reference: [Checking disk usage with ncdu](https://ostechnix.com/check-disk-space-usage-linux-using-ncdu/).

### 2.7 Storage housekeeping: do and do not

- Check your home-directory disk usage regularly.
- Transfer and back up your data to your personal device.
- Remove unwanted data from your own home directory.
- **Do not delete system configuration files or directories just because they begin with a dot**, for example `.bash_profile`, `.bashrc`, and `.config`.
- `.conda` stores packages and environments you have set up for computation. Remove **unwanted environments** to free space; see the targeted environment-removal instructions below.
- The cluster provides **no backup service**. You are responsible for protecting and maintaining your data.

## 3. Conda packages, environments, and CUDA

*Original pages 7–13.*

### 3.1 Environment modules

TC1 provides the `module` tool to configure the applications available in a shell or job. The cluster is shared across different CCDS undergraduate courses and projects, so some installed applications may not be relevant to your work.

Shared application modules are under `/cm/shared/modulefiles`. The screenshots also show `/cm/local/modulefiles` for cluster/system tools.

```bash
# View available applications/modules
module avail

# View the description and configuration of a module
module show <module_name>

# Load a module
module load <module_name>

# List loaded modules
module list

# Unload a selected module
module unload <module_name>

# Unload all loaded modules
module purge

# Help
module --help
```

**Important:** the `slurm` module should be loaded upon every login. If it is missing, load it manually; it provides the SLURM commands needed to submit and manage jobs:

```bash
module load slurm
module list
```

The example output shows `slurm/slurm/23.02.8`. Treat screenshot versions as examples, and use `module avail` to inspect what is actually available.

For completeness, the page 7 screenshot shows this historical module inventory:

```text
/cm/local/modulefiles
  boost/1.81.0                  cluster-tools/10.0
  cm-bios-tools                 cm-image/10.0
  cm-scale/cm-scale.module      cm-setup/10.0
  cmd                           cmjob
  cmsh                          docker/26.1.5
  dot                           freeipmi/1.6.14
  gcc/13.1.0                    ipmitool/1.8.19
  lua/5.4.6                     luajit
  mariadb-libs                  module-git
  module-info                   null
  openldap                      python3
  python39                      rocm-smi/4.3.0
  shared                        slurm/slurm/23.02.8

/cm/shared/modulefiles
  anaconda                      blas/gcc/64/3.11.0
  bonnie++/2.00a                cm-pmix3/3.1.7
  cm-pmix4/4.1.3                cuda/11.8
  cuda/12.2                     cuda/12.4
  cuda/12.5                     cuda/12.6
  cuda/12.9                     default-environment
  gcc/11.5.0                    gcc/12.4.0
  gcc/13.3.0                    gcc/14.1.0
  gdb/13.1                      globalarrays/openmpi/gcc/64/5.8
  hdf5/1.14.0                   hdf5_18/1.8.21
  hwloc/1.11.13                 hwloc2/2.8.0
  iozone/3.494                  lapack/gcc/64/3.11.0
  miniconda/py39                miniconda/py312
  miniconda/py313               mpich/ge/gcc/64/4.1.1
  mvapich2/gcc/64/2.3.7          netcdf/gcc/64/gcc/64/4.9.2
  netperf/2.7.0                 openblas/dynamic (default)
  openblas/dynamic/0.3.18        openmpi/gcc/64/4.1.5
  openmpi4/gcc/4.1.5            python/3.9.23
  python/3.10.18                python/3.11.13
  python/3.12.11                python/3.13.5
  ucx/1.10.1
```

### 3.2 Permissions and choosing a Conda installation

Users **cannot use `sudo` or install applications into the system**. Use the provided Anaconda/Miniconda module and your own Conda environments.

Source background links:

- [Anaconda Distribution](https://www.anaconda.com/products/distribution)
- [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html)

Load the available Anaconda module:

```bash
module load anaconda
```

Find the Conda executable using either command:

```bash
whereis conda
which conda
```

Screenshots show paths such as `/tc1apps/anaconda3/bin/conda` and `/tc1share1/tc1apps/anaconda3/bin/conda`. Use the path reported on the cluster rather than hard-coding a screenshot path.

### 3.3 First-time Conda initialization

Initialize Conda for Bash:

```bash
conda init bash
```

This adds a Conda initialization block to `~/.bashrc`. On subsequent logins, the Conda base environment may load automatically and appear as `(base)` at the prompt.

Apply the change by either logging out and logging in again, or sourcing your `.bashrc` in the current shell:

```bash
source ~/.bashrc
```

The source uses `source .bashrc` while in the home directory. The explicit `~/` form above identifies the same home-directory file regardless of the current directory.

Useful inspection and exit commands:

```bash
# Information about the loaded Conda installation
conda info

# Packages in the active environment, including base when base is active
conda list

# Leave the active Conda environment
conda deactivate
```

### 3.4 Switching Anaconda/Miniconda modules

Apply the same initialization approach when using an available Miniconda module. **Deactivate the existing Conda environment before loading another Conda module.**

The source's Python 3.9 Miniconda example is:

```bash
conda deactivate
module purge
module load miniconda/py39
```

The screenshots then locate the new `conda`, initialize Bash, source `.bashrc`, and check Python:

```bash
whereis conda
conda init bash
source ~/.bashrc
python -V
```

The sample Miniconda path is `/tc1share1/tc1apps/miniconda/py39/bin/conda`, and the sample Python version is `3.9.19`.

**Remember:** `module purge` unloads all modules, including `slurm`; load `slurm` again before using scheduler commands if necessary.

### 3.5 Always create your own environment

**You are not allowed to install packages in the shared base environment.** If its packages do not meet your requirements, create an environment in your home directory and install the packages there.

Named environments are created by default under `~/.conda/envs/` for the user. The source writes this as `~username/.conda/envs/`.

Create and activate `TestEnv`:

```bash
conda create -n TestEnv
conda activate TestEnv
```

Create and activate an environment at a custom location:

```bash
conda create --prefix /scratch-shared/myteam/envs/TestEnv
conda activate /scratch-shared/myteam/envs/TestEnv
```

`/scratch-shared/myteam/envs` is the source's illustrative custom path. Use a custom path only if it is within storage assigned/authorised to you.

Leave an environment or list your environments:

```bash
conda deactivate
conda env list
```

### 3.6 Clone, export, restore, update, or remove environments

The source describes renaming an environment by **cloning it and removing the old environment**:

```bash
conda create -n NewEnv --clone OldEnv
conda env remove -n OldEnv
```

Export the **active** environment configuration to a YAML file for backup:

```bash
conda env export > TestEnv.yml
```

Create/restore an environment named `Env002` from that file:

```bash
conda env create -n Env002 -f TestEnv.yml
```

Update a prefix-based environment from YAML, pruning outdated dependencies:

```bash
conda env update --prefix ./TestEnv --file TestEnv.yml --prune
```

This applies the YAML specification **to the environment** at `./TestEnv`; it does not rewrite the YAML file. The source's description of this command is clarified in the conversion notes.

Remove `TestEnv` when it is no longer needed:

```bash
conda env remove -n TestEnv
```

Reference: [Managing Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

### 3.7 Environment variables

List variables configured for the active environment:

```bash
conda env config vars list
```

Set or unset a variable:

```bash
conda env config vars set my_var1=value
conda env config vars unset my_var1
```

**Reactivate the environment after adding or removing a variable.** The screenshot demonstrates the complete sequence with the value `10`:

```bash
conda activate TestEnv
conda env config vars list
conda env config vars set my_var1=10
conda deactivate
conda activate TestEnv
conda env config vars list
# Example output: my_var1 = 10

conda env config vars unset my_var1
conda deactivate
conda activate TestEnv
conda env config vars list
```

### 3.8 Installing and managing packages

Always create and activate an environment of your own first:

```bash
conda create -n TestEnv
conda activate TestEnv
```

Search for Python and install a selected version:

```bash
conda search python
conda install python=3.11
```

Conda normally uses its configured default channels. Specify a channel when searching/installing from a third-party source. The guide uses **conda-forge**, described as a source of newer Conda packages:

```bash
conda search -c conda-forge <package_name>
conda install -c conda-forge <package_name>
```

An alternative channel-qualified install syntax is:

```bash
conda install conda-forge::<package_name>
```

List installed packages:

```bash
conda list
```

Install another selected version, replacing the existing version in the environment if necessary:

```bash
conda install python=3.10
```

The source's package-removal example is:

```bash
conda uninstall python=3.9
```

**Important source advice:** use `conda install` whenever the required package is available through Conda. Use `pip install` only if it is not, to reduce the chance of incompatible packages.

References:

- [Conda channels](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html)
- [Installing Anaconda packages](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/)
- [Managing packages with Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages)

### 3.9 Removing unwanted environments to free storage

1. Leave the environment:

   ```bash
   conda deactivate
   ```

2. List the available environments and identify the unwanted environment in **your home directory**:

   ```bash
   conda env list
   ```

3. Remove only the unwanted environment you own:

   ```bash
   conda env remove --name <environment_name>
   ```

4. The source also permits deleting the directory of an already **deactivated**, unwanted environment under your own `.conda/envs` directory. Its screenshot removes `/home/UG/scetest5/.conda/envs/TestEnv`. The same example expressed relative to your home is:

   ```bash
   rm -Rf ~/.conda/envs/TestEnv
   ```

   This deletes that environment directory and its contents. Confirm the exact environment and path first. Do not delete the shared base installation or unrelated dotfiles/directories.

### 3.10 CUDA choices in TC1

**CUDA — Compute Unified Device Architecture** — is NVIDIA's software framework for GPU acceleration. TC1 offers two approaches:

- **System-installed CUDA:** available on compute nodes through modules loaded in the job script.
- **Conda CUDA:** the required toolkit or CUDA-related packages installed in your own Conda environment.

The NVIDIA GPUs are in compute nodes named `TC1Nxx`. They are **not** in the head node `CCDS-TC1`. Do not run GPU/CUDA hardware checks on the head node.

List CUDA modules:

```bash
module avail | grep cuda
```

The screenshots show examples including `cuda/11.8`, `cuda/12.2`, `cuda/12.4`, `cuda/12.5`, `cuda/12.6`, and `cuda/12.9`.

Load the selected system CUDA in a job script, for example:

```bash
module load cuda/12.9
```

The available system CUDA may not meet every user's requirements. In that case, create your own Conda environment and install the appropriate toolkit/packages there.

Source CUDA/framework references:

- [NVIDIA CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Previous PyTorch versions](https://pytorch.org/get-started/previous-versions/)
- [TensorFlow tested build configurations](https://www.tensorflow.org/install/source#tested_build_configurations)
- [Anaconda cudatoolkit package](https://anaconda.org/anaconda/cudatoolkit)

### 3.11 Installing CUDA packages into your own environment

After activating your own environment, search for an appropriate version:

```bash
# NVIDIA channel toolkit package
conda search -c nvidia cuda-toolkit

# Anaconda/default-channel toolkit package
conda search cudatoolkit
```

Install the selected toolkit or NVIDIA CUDA compiler (`nvcc`), as needed:

```bash
conda install -c nvidia cuda-toolkit=<version>
conda install -c nvidia cuda-nvcc=<version>
```

### 3.12 What can be checked on the head node

The source explicitly permits checking **installed package records** in your environment on the head node:

```bash
conda activate <environment_name>
conda list | grep cuda
```

This checks what packages are installed. To test actual GPU availability, the toolkit/compiler in a compute-node session, or framework/GPU operation, use an allocated **compute node**.

### 3.13 Launch a compute-node Jupyter session to verify the environment

Create a SLURM script and submit it to launch JupyterLab, then open its terminal. Full setup and tunnelling instructions appear in sections 6–7.

The screenshot on original page 12 contains this complete shared-environment example:

```bash
#!/bin/sh
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --mem=10G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=TC1N04
#SBATCH --time=360
#SBATCH --job-name=J1
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
source activate /tc1apps/2_conda_env/Jupyter
jupyter-lab --ip=$(hostname -i) --port=8882
```

The node, partition, QoS, resource requests, and port are source examples. Adapt them to your assignment and availability.

In JupyterLab, select **Terminal** under **Other** to start a terminal session on the allocated compute node. List and activate your own environments:

```bash
conda env list
conda activate <environment_name>
```

The example lists `RunJupyter`, `cuda-python`, `myCZ4042`, `numba-test`, and `tf-gpu` under `/home/UG/scetest5/.conda/envs/`, plus the base environment at `/tc1apps/anaconda3`. It demonstrates `conda activate numba-test`.

### 3.14 Verify PyTorch and Conda CUDA on the compute node

The screenshot lists shared environments and activates the one containing PyTorch, PyTorch-CUDA, and CUDA-NVCC:

```bash
ll /tc1apps/2_conda_env/
conda activate /tc1apps/2_conda_env/CZ4042_v6
conda list | grep cuda
python
```

At the Python prompt:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
quit()
```

The example outputs are:

```text
True
11.8
```

The example Python version is `3.12.3`, and the package listing includes PyTorch `2.5.1`, PyTorch-CUDA `11.8`, CUDA-NVCC `11.8.89`, and related CUDA runtime/library packages. It also shows a `cuda-version` package of `12.8`; retain the distinction between individual package metadata and the `torch.version.cuda` value.

The screenshot's package/version pairs are:

```text
cuda-cudart       11.8.89
cuda-cupti        11.8.87
cuda-libraries    11.8.0
cuda-nvcc         11.8.89
cuda-nvrtc        11.8.89
cuda-nvtx         11.8.86
cuda-runtime      11.8.0
cuda-version      12.8
pytorch           2.5.1  (build: py3.12_cuda11.8_cudnn9.1.0_0)
pytorch-cuda      11.8
pytorch-mutex     1.0    (cuda)
```

Then, still on the compute node:

```bash
nvcc -V
```

The sample output reports CUDA compiler release `11.8`, `V11.8.89`.

### 3.15 Verify system-installed CUDA on the compute node

The screenshot uses:

```bash
module list
module avail | grep cuda
module load cuda/12.4
module list
nvcc -V
nvidia-smi
```

The compiler example reports release `12.4`, `V12.4.131`. The `nvidia-smi` example reports a Tesla V100-PCIE-32GB, `32768 MiB` memory, and no running processes at that moment; the screenshot uses NVIDIA driver `575.57.08`.

**Important:** `nvidia-smi` shows information only for the GPU assigned to you in that allocation.

### 3.16 Verify a CUDA-Python environment on the compute node

The source's CUDA-Python example uses:

```bash
conda activate cuda-python
conda list | grep cuda-python
conda list | grep nvcc
python -V
nvcc -V
which nvcc
```

The screenshot shows:

- `cuda-python` version `11.8.0` from NVIDIA.
- Compiler-related packages `cuda-nvcc`, `cuda-nvcc-dev_linux-64`, `cuda-nvcc-impl`, `cuda-nvcc-tools`, and `cuda-nvcc_linux-64`, version `12.4.131`.
- Python `3.9.19`.
- `nvcc` release `12.4`, `V12.4.131`.
- The environment-local executable `/home/UG/scetest5/.conda/envs/cuda-python/bin/nvcc`.

Reference: [CUDA Python documentation](https://nvidia.github.io/cuda-python/latest/).

## 4. SLURM user guide

*Original pages 14–15.*

SLURM submits, schedules, and monitors jobs on computer clusters. In TC1, write a job script specifying the resources, applications, and execution paths, then submit it from the command line. SLURM allocates a compute node; **do not compile or execute your computation on the head node**.

### 4.1 Find your assigned QoS

Replace `<username>` with your own account ID:

```bash
sacctmgr show user <username> withassoc format=user,qos
```

The source example shows user `scetest5` assigned the QoS `normal`.

Inspect a QoS's configured resources using either output format:

```bash
sacctmgr -P show qos <qos_name> withassoc format=name,MaxTRESPU,MaxJobsPU,MaxWall

sacctmgr show qos <qos_name> withassoc format=name%+15,MaxTRESPU%+40,MaxJobsPU%+10,MaxWall%+10
```

The `%+<value>` notation expands the display field width. The source's `normal` example is:

```text
Name:       normal
MaxTRESPU:  cpu=20,gres/gpu=1,mem=64G
MaxJobsPU:  2
MaxWall:    06:00:00
```

Definitions:

- **MaxTRESPU — Maximum Trackable RESources Per User:** the user's maximum CPU/core, GPU, and memory resource allocation. The example allows 20 CPU cores, one GPU, and 64 GB of memory.
- **MaxJobsPU:** the maximum number of a user's jobs that can run at a given time. The example is two.
- **MaxWall:** maximum wall-clock time, shown in a day/hour/minute/second format. The example is six hours.

These values are examples from the source, not a guarantee for your account. The account-specific `MyTCinfo` command is described in section 5.

SLURM job-script references:

- [SLURM quick start](https://slurm.schedmd.com/quickstart.html)
- [MIT Svante SLURM guide](https://svante.mit.edu/use_slurm.html)

### 4.2 Inspect a job or node with scontrol

View detailed information for a running job, including the allocated node and GPU index:

```bash
scontrol show -d jobid <jobid>
```

The equivalent option ordering used later in the guide is:

```bash
scontrol -d show jobid <jobid>
```

Inspect the compute nodes:

```bash
scontrol show node TC1N[01-07]
```

Compare these node fields:

- **CfgTRES:** configured trackable resources.
- **AllocTRES:** resources currently allocated to running jobs on that node.

The source mentions the older partition label `CCDSGPU_UG` here, while the main submission examples use `UGGPU-TC1`. Check the available partition as described below.

### 4.3 Inspect node states with sinfo

```bash
sinfo -N -l
```

This displays compute-node states by partition. The screenshot lists `TC1N01` through `TC1N07` in `UGGPU-TC1`, with a mixture of `idle` and `mixed` states. Its example nodes each show 72 CPUs and memory values around `386590` in the displayed units.

The source advises that a state other than `idle` or `mix` may indicate a node is out of service. See section 5.7 for its more detailed state explanations, including temporary states.

### 4.4 Inspect job history with sacct

Your own history, with selected fields:

```bash
sacct --format=jobid,jobname%+15,qos,alloctres%+50,nodelist,start,elapsed,state,reason%+20
```

All users' history for the day, as shown in the source:

```bash
sacct -a --format=user,jobid,jobname%+15,qos,alloctres%+50,nodelist,start,elapsed,state,reason%+20
```

Filter out lines containing the word `batch`:

```bash
sacct -a --format=user,jobid,jobname%+15,qos,alloctres%+50,nodelist,start,elapsed,state,reason%+20 | grep -wv batch
```

List a specified user's jobs:

```bash
sacct -u <username>
```

### 4.5 Submit, inspect, or cancel jobs

Submit a script for background execution:

```bash
sbatch job.sh
```

Once an ordinary batch job is submitted, you may leave the SSH session without affecting that job. Jupyter's browser access additionally requires a maintained SSH tunnel, as explained later.

Display queued jobs:

```bash
squeue
```

Display more information:

```bash
squeue -la
```

Display a specified user's jobs:

```bash
squeue -u <username>
```

Obtain a job's ID from `squeue`, then cancel it:

```bash
scancel <jobid>
```

### 4.6 Inspect completed-job efficiency

```bash
seff <completed_jobid>
```

The **SLURM Job Efficiency Report** shows resource-use statistics for a completed job. Use it to review and improve the resources requested by later jobs.

## 5. Guidelines for job submission

*Original pages 16–22.*

### 5.1 Do not execute your computation on the head node

**Do not run your code on `CCDS-TC1`.** Doing so can cause slow or failed SSH access, frozen file browsing, and service suspension affecting other users.

Processes found running computation with high CPU/memory usage on the head node may be **terminated without prior notification**. **Repeated offenders can be banned from TC1.**

### 5.2 Use only your assigned QoS

A job submitted using an unauthorised QoS can be terminated by the system **without prior notification**. Use only a QoS assigned to your account and allowed in the selected partition.

### 5.3 Check account limits and apply them across your jobs

The cluster provides a customised account-information command:

```bash
MyTCinfo
```

It displays account information, home directory, SLURM association, QoS limits, and partition information. The source's example shows:

```text
User:       scetest5
Home:       /home/UG/scetest5
Cluster:    slurm
QoS:        normal
MaxTRESPU:  cpu=20,gres/gpu=1,mem=64G
MaxSubmit:  2
MaxWall:    06:00:00
Partition:  UGGPU-TC1
```

The source explains its example limits as follows:

- **MaxSubmit ≤ 2:** at most two jobs may be submitted at a time. You may run two jobs if the resources are assigned correctly across them.
- **MaxWall ≤ 6 hours:** the source says the total computation of running jobs must not exceed the assigned maximum wall time; it also checks each job's requested time against `MaxWall` elsewhere. Preserve both statements when reviewing your assignment; the time flag must not exceed your allowed wall time.
- **MaxTRESPU:** the total resource requests across your submitted/running jobs using the QoS must remain within the per-user maximum. In the example, CPU cores total at most 20, memory totals at most 64 GB, and only one GPU is available to the user.

**QoS values may change each school semester.** Always inspect your actual assignment.

Other customised scripts are available under:

```text
/tc1share2/tc-scripts
```

The screenshot lists `MyJobHistory`, `MyTCinfo`, `TC1JobHistory`, `TC1RunningJob`, and `TC1userinfo`, plus a notice file named `1_The_scripts_can_only_run_in_HeadNode`. These are shown as **head-node tools** in the source.

The screenshot demonstrates listing current activity with:

```bash
TC1RunningJob
```

Its explanatory output says each node has **three GPU cards**, each job can use **one GPU**, and therefore at most **three GPU jobs per node** can run concurrently under that configuration. GPU card sequence labels are `[1]-IDX:0`, `[2]-IDX:1`, and `[3]-IDX:2`.

### 5.4 Create and inspect a plain-text job script

Create the job script using an editor such as Windows Notepad, Sublime Text on macOS, or `gedit` on Linux. Save it with a **`.sh`** extension and transfer it to your home directory on the cluster.

**A filename extension does not establish the actual file format in Linux.** Inspect the file using the `file` command, for example:

```bash
file job.sh
```

Reference: [Linux file command](https://www.linuxfordevices.com/tutorials/linux/file-command-in-linux).

After transferring a script to TC1, inspect its contents before submitting it. A common submission failure is invalid/unknown characters introduced into the script. The source screenshot shows boxes replacing the two ASCII hyphens `--` in SLURM options and command arguments.

If necessary, edit the file directly on the head node using **`vi`**, **`nano`**, or **`ne`** to correct the characters.

Editor references:

- [vi editor guide](https://www.guru99.com/the-vi-editor.html)
- [nano text editor](https://www.geeksforgeeks.org/nano-text-editor-in-linux/)
- [Linux command-line text editors](https://itsfoss.com/command-line-text-editors-linux/)

### 5.5 Job-script resource flags

The source marks partition, QoS, GPU, memory, node count, time, job name, output, and error flags with an asterisk as required, while explicitly allowing omission of the GPU flag for CPU-only work.

#### Partition — required

```bash
#SBATCH --partition=UGGPU-TC1
```

A partition identifies the node group for the job. List partitions and the QoS allowed for them:

```bash
scontrol show partition
```

#### QoS — required

```bash
#SBATCH --qos=normal
```

Choose a QoS assigned to you and permitted in the selected partition.

#### GPU — include for GPU computation

```bash
#SBATCH --gres=gpu:1
```

This requests a GPU on the compute node. The GPU count must comply with your QoS. **Omit this flag if your computation does not use a GPU.**

#### System memory — required

Either example is shown in the source:

```bash
#SBATCH --mem=8000M
```

or:

```bash
#SBATCH --mem=8G
```

Specify the memory required for the job, within your QoS maximum. The source says this option must be present so memory can be allocated. It is a system-memory request; the separate GPU-memory limitation is explained below.

#### CPU tasks/cores — optional, defaults of one

For more tasks per node, associated with distributed parallelism:

```bash
#SBATCH --ntasks-per-node=2
```

For more CPU cores per task, associated with multithreaded tasks:

```bash
#SBATCH --cpus-per-task=2
```

The source prints the second option as `--cpus-per-tasks=2`; this Markdown uses the corrected singular form `--cpus-per-task=2`. Both options default to one in the guide. Add them only when your computation needs additional CPU resources and keep the overall allocation within your QoS.

#### Number of nodes — required

```bash
#SBATCH --nodes=1
```

**The maximum node count allowed per job in the source is one.**

#### Select a particular node — optional

```bash
#SBATCH --nodelist=TC1Nxx
```

Replace `xx` with `01`, `02`, `03`, `04`, `05`, `06`, or `07`.

List idle nodes:

```bash
sinfo -N | grep idle
```

If you omit `--nodelist`, SLURM chooses a ready node. The source advises selecting an idle node explicitly only if the job requires high CPU or memory.

#### Time limit — required

```bash
#SBATCH --time=60
```

Supported source examples for the format are:

```text
<minutes>
<minutes>:<seconds>
<hours>:<minutes>:<seconds>
<days>-<hours>:<minutes>:<seconds>
<days>-<hours>
```

In the sample scripts, `60` means 60 minutes and `360` means six hours. Stay within your assigned wall-time limit.

#### Job name — required

```bash
#SBATCH --job-name=MyJob
```

#### Standard-output file — required

Choose a filename pattern, for example:

```bash
#SBATCH --output=output_%j.out
```

or:

```bash
#SBATCH --output=output_%x_%j.out
```

#### Error file — required

```bash
#SBATCH --error=error_%j.err
```

or:

```bash
#SBATCH --error=error_%x_%j.err
```

Filename substitutions:

- `%x` is the job name.
- `%j` is the job ID generated by SLURM.

#### Complete standard script: job.sh

The original page 18 example is:

```bash
#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=60
#SBATCH --job-name=MyJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/12.2
module load anaconda
source activate TestEnv
python test.py
```

The last four lines load the necessary modules, activate your environment, and execute your code. Adjust the environment, script path, software modules, and resource requests for your computation. Add other necessary flags from the preceding guidance. A job requesting more than its QoS permits will not run.

The script produces separate standard-output and error files:

```text
output_MyJob_<jobid>.out
error_MyJob_<jobid>.err
```

#### Can GPU memory be adjusted in the job script?

The source identifies TC1's cards as **NVIDIA Tesla V100 32 GB**. Their installed GPU memory is fixed at **32 GB** and cannot be resized by a job-script flag. Manage the desired GPU-memory usage in your code; the source suggests looking up framework-specific guidance.

- [TensorFlow GPU guide](https://www.tensorflow.org/guide/gpu)
- [Managing GPU memory with TensorFlow and PyTorch](https://wiki.ncsa.illinois.edu/display/ISL20/Managing+GPU+memory+when+using+Tensorflow+and+Pytorch)

### 5.6 Submit with sbatch; avoid srun for this workflow

Submit the script:

```bash
sbatch job.sh
```

SLURM responds with a job ID, for example:

```text
Submitted batch job 120
```

The screenshot submits `hellosubmit.sh`; `120` is its example ID.

The source advises **avoiding `srun` to submit jobs** in this workflow:

- `srun` submits for real-time execution from the command line and requires the SSH session to remain connected for the process.
- Disconnection may kill the process or cause loss of control of execution.
- Computations lasting more than one hour are specifically advised to use `sbatch`.
- `sbatch` submits work to SLURM for background handling; the user can disconnect after submission and return later for results.
- Using `sbatch` and leaving the session also reduces the volume of SSH connections to the head node.

**Jupyter exception:** the job can be submitted with `sbatch`, but its browser access needs a maintained SSH tunnel. See sections 6–7.

### 5.7 Check node operational status

```bash
sinfo
```

Reference: [SLURM sinfo documentation](https://slurm.schedmd.com/sinfo.html).

States explained by the source:

- **`idle`:** the node is alive and free to accept a job.
- **`mixed`:** jobs are running on the node.
- **`down`:** the SLURM service on the node is down.
- **`completing`:** the node is completing the final job computation.
- **`mixed+drain` / `draining`:** jobs are still running, but failed tasks are unable to terminate or are still terminating. Performance can degrade, making other jobs slow or hang. If the administrator resets the node, its running jobs are terminated; the source says affected jobs join the pending queue for reassignment.
- **`drained`:** the node has a failed task and cannot run jobs; an administrator must inspect and reset it.

The screenshot also shows states such as `down*` with **Not responding** in the reason field.

Cluster status logs are under:

```text
/tc1share2/Log-TC1Status
```

Active-node filenames use:

```text
ActiveNode_YYYY-MM-DD-HH-mm
```

Examples from the source:

```bash
# Active nodes with running jobs at 3 p.m. on 14 January 2025
more /tc1share2/Log-TC1Status/ActiveNode_2025-01-14-15-*

# Active-node logs for 1 January 2025
more /tc1share2/Log-TC1Status/ActiveNode_2025-01-01-*

# Overall cluster status
more /tc1share2/Log-TC1Status/TC1_ClusterInfo
```

The dates are historical examples; substitute a date whose logs are available.

### 5.8 Check current jobs and diagnose pending jobs

```bash
squeue
squeue -la
sacct -u <username>
scontrol show jobid <jobid>
```

These respectively show the queue, a more detailed queue view, the user's job history, and details of a running/pending job.

Reference: [SLURM squeue documentation](https://slurm.schedmd.com/squeue.html).

In `squeue`:

- **`ST`** is the job-state column.
- **`PD`** means pending.
- **`R`** means running.
- **`NODELIST (REASON)`** shows the assigned node for a running job, or the reason a job is pending.

The source lists these causes of jobs waiting or failing to run:

- Requested resources exceed the QoS limit.
- TC1 cannot allocate the requested node/resources.
- The job was interrupted by system failure or resource shortage and returned to pending.

Resources mean the compute nodes and their available CPU, memory, and GPU cards. The source describes allocation as **first come, first served**, using the resource values in the job script. When resources are available and the code has no execution issue, a job runs with state `R` on its assigned node. If resources cannot be allocated, it remains `PD`.

#### QOSMaxWallDurationPerJobLimit

Check the job's time request:

```bash
#SBATCH --time=<allowed_time>
```

It must not exceed the QoS `MaxWall`.

#### QOSMaxJobsPerUserLimit

Check your total job count against `MaxJobsPU`. Subsequent jobs may enter the pending queue or be rejected when the limit is exceeded.

#### QOSMaxMemoryPerUser, QOSMaxCpuPerUserLimit, QOSMaxGRESPerUser

Check total resource requests across jobs using the same QoS against `MaxTRESPU`. The source specifically points to `--mem` and `--ntasks-per-node`, as well as the GPU request.

#### QOSMaxGRESPerUser when the user has one GPU

The source says one allocated GPU runs one job at a time. If your QoS permits `gres/gpu=1`, a subsequent GPU job using that QoS cannot obtain another GPU concurrently.

If the subsequent computation is CPU-only, omit:

```bash
#SBATCH --gres=gpu:1
```

#### Priority

The job is waiting on scheduling priority and can run when the required node/resources become available. The source describes it as waiting with higher priority.

#### Resources

The job is waiting for available resources and can run when those resources are allocated.

The source says interrupted jobs may return to pending with `Priority` or `Resources` as the reason. It also states that a pending job will be removed if its waiting time exceeds a time limit referenced to the QoS `MaxWall`; this is preserved as a source statement, not independently verified scheduler behaviour.

#### What to do about a job stuck pending

1. Cancel pending jobs that fail the QoS-limit check.
2. Review the script and reduce/adjust CPU and memory requests to acceptable values that can fit an available node.
3. Restrict the number of jobs submitted.
4. Resubmit a corrected job when appropriate.

Example cancellation, where `263` is the job ID:

```bash
scancel 263
```

### 5.9 Read results and inspect job history

When a job completes, it disappears from the active queue. Inspect its output files for results.

Unless a different output path is specified in the job script, the output files are created **in the directory from which the job was submitted**.

With these directives:

```bash
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
```

the example names are `output_MyJob_<jobid>.out` and `error_MyJob_<jobid>.err`. The page 21 screenshot also shows the concrete pair `output_J1_114.out` and `error_J1_114.err` alongside `J1.sh`.

View file contents with **`cat`**, **`less`**, **`more`**, or **`tail`**.

Reference: [Unix/Linux commands to view files](https://www.cyberciti.biz/faq/unix-linux-command-to-view-file/).

For the day's detailed job history, use:

```bash
MyJobHistory
```

The customised report includes:

- Applied QoS.
- Allocated CPU, GPU, and memory resources.
- Assigned node.
- Submission time and start time.
- Elapsed time.
- State and reason.

### 5.10 Use seff to improve CPU and memory requests

For a completed job:

```bash
seff <jobid>
```

Review the actual CPU/core and memory use, then adjust resource flags for subsequent runs. The source's screenshots show why both underuse and overuse need attention:

- Example job `115`: completed with exit code 0, one node, four cores per node, CPU efficiency `0.00%` of `00:00:24` core-walltime, job wall-clock time `00:00:06`, memory used `1.32 MB`, and memory efficiency `0.00%` of `16.00 GB`.
- Another completed job: one core, CPU time `00:02:16`, CPU efficiency `144.68%` of `00:01:34` core-walltime, memory used `2.54 GB`, and memory efficiency `25.40%` of `10.00 GB`.
- Another completed job: one core, CPU time `00:44:59`, CPU efficiency `103.05%` of `00:43:39` core-walltime, memory used `278.29 GB`, and memory efficiency `434.83%` of `64.00 GB`.

These are screenshot examples, not recommended requests. The source advises reviewing both the code and the CPU/memory settings when utilisation is high.

### 5.11 Identify the GPU allocated to a running job

```bash
scontrol -d show jobid <running_jobid>
```

Look for details such as:

```text
NodeList=TC1N04
JOB_GRES=gpu:1
Nodes=TC1N04 ... GRES=gpu:1(IDX:0)
```

Each compute node's GPU cards are indexed by SLURM as `IDX:0`, `IDX:1`, and `IDX:2`. The example above represents **the first GPU card (`IDX:0`) on `TC1N04`**. The GPU count in `JOB_GRES=gpu:1` and the card index in `IDX:0` are different pieces of information.

### 5.12 GPU-utilisation metrics and temperature warning

The source explains that cluster-wide GPU utilisation is retrieved from active compute nodes using `nvidia-smi` and requires root access for that collection. Users can consult the collected logs. Within your allocated compute-node session, `nvidia-smi` shows the GPU assigned to you, as described earlier.

The GPU query fields documented in the source are:

- **`name`:** official GPU product name.
- **`pci.bus_id`:** PCI bus ID.
- **`driver_version`:** installed NVIDIA display-driver version.
- **`pstate`:** current performance state, from `P0` for maximum performance to `P12` for minimum performance.
- **`power.limit` [W]:** maximum power the GPU can draw.
- **`power.draw` [W]:** current power draw.
- **`temperature.gpu` [°C]:** GPU-core temperature.
- **`memory.total` [MiB]:** installed GPU memory.
- **`memory.used` [MiB]:** memory allocated by active contexts.
- **`utilization.gpu` [%]:** percentage of the preceding second during which at least one kernel executed on the GPU.
- **`utilization.memory` [%]:** percentage of the preceding second during which global/device memory was read or written.

The utilisation rates indicate how busy the GPU is over time and how much the assigned job uses it.

The example log lists three Tesla V100-PCIE-32GB cards with bus IDs `00000000:37:00.0`, `00000000:86:00.0`, and `00000000:D8:00.0`, driver `565.57.01`, and total memory `32768 MiB` each. The first shows active usage while the other two show low/no usage; the values are a historical snapshot.

The remaining values in that snapshot are:

```text
Bus ID              State  Power limit  Power draw  Temp  Memory used  GPU util  Memory util
00000000:37:00.0     P0     250.00 W     145.42 W    72 C  3594 MiB     95%       30%
00000000:86:00.0     P0     250.00 W      35.20 W    29 C     1 MiB      0%        0%
00000000:D8:00.0     P0     250.00 W      35.65 W    35 C     1 MiB      0%        0%
```

**Temperature warning from the source:** optimal GPU temperature should remain below **85°C**. If it reaches **90°C** while GPU utilisation remains at **100%**, the source calls for urgent action to terminate computation on the affected node. Users should cancel their own affected job; node-wide administration belongs to the administrator.

### 5.13 Collected cluster logs

Usage records are captured on a schedule and retained in these locations:

#### Running-job history: past 24 hours

```text
Directory: /tc1share2/Log-RunHistory
Filename:  Running_YYYY-MM-DD-HH-mm
```

Records detected running jobs, showing the actual GPU used by a job at **five-minute intervals**.

#### Job history: past 48 hours

```text
Directory: /tc1share2/Log-JobHistory
Filename:  JobHistory_YYYY-MM-DD-HH-mm
```

Records jobs computed during that period.

#### Cluster status: past 28 hours

```text
Directory: /tc1share2/Log-TC1Status
Files:     TC1_ClusterInfo
           ActiveNode_YYYY-MM-DD-HH-mm
```

Records overall TC1 resource utilisation.

The source's example for viewing a day's job logs is:

```bash
more /tc1share2/Log-JobHistory/JobHistory_2025-01-09-*
```

Use dates actually present in the retained logs.

### 5.14 Allocate resources reasonably

The source describes resource assignment as first come, first served, based on each job script's flags and values. Jobs arriving later use the resources left over. Requesting excessive resources therefore reduces what is available to others.

The guide identifies two extremes:

- Performance degradation when utilisation exceeds **100%** of the allocation.
- Wastage when utilisation remains below **10%** of the allocation.

Use `seff` to review CPU and memory use, then set reasonable values for later jobs. Use the `/tc1share2/Log-*` records to review whether past computations used their allocated GPU reasonably.

## 6. Shared resources

*Original page 23.*

### 6.1 Ready-made shared Conda environments

The source lists shared environments under:

```text
/tc1apps/2_conda_env/
```

#### CZ4042_v4

Path:

```text
/tc1apps/2_conda_env/CZ4042_v4
```

Applications recorded in the source:

- Python `3.10`
- PyTorch `2.4`
- PyTorch-CUDA `12.4`
- cuDNN `8.9`
- TensorFlow `2.9`
- CUDA Toolkit `11.8`

The source records this environment variable for TensorFlow 2.9:

```text
TF_ENABLE_ONEDNN_OPTS=0
```

It gives the configuration command:

```bash
conda env config vars set TF_ENABLE_ONEDNN_OPTS=0
```

The shared environment is described as already configured. Make changes for your own work in an environment you own.

#### CZ4042_v5

Path:

```text
/tc1apps/2_conda_env/CZ4042_v5
```

Applications recorded in the source:

- Python `3.11.4`
- PyTorch `2.0.1`
- PyTorch-CUDA `11.7`
- Jupyter

#### CZ4042_v6

Path:

```text
/tc1apps/2_conda_env/CZ4042_v6
```

Applications recorded in the source:

- Python `3.12.3`
- PyTorch `2.5.1`
- PyTorch-CUDA `11.8`
- Jupyter

The February 2026 guide describes this as the latest environment created for **SC4001/CZ4042**.

An exported environment specification is available at:

```text
/tc1apps/3_conda_yml/SC4001.yml
```

Copy it into your home directory and create a similar environment named `mySC4001`:

```bash
cd
cp /tc1apps/3_conda_yml/SC4001.yml .
conda env create -f SC4001.yml -n mySC4001
```

#### Jupyter

Path:

```text
/tc1apps/2_conda_env/Jupyter
```

Applications recorded in the source:

- Python `3.12`
- Jupyter, including **JupyterLab and Notebook**

### 6.2 Launch JupyterLab from the shared environment

Include these commands **inside a SLURM job script** with your appropriate resource directives:

```bash
module load anaconda
source activate /tc1apps/2_conda_env/Jupyter
jupyter-lab --ip=$(hostname -i) --port=8881
```

After the job launches Jupyter, set up SSH tunnelling for browser access. The port `8881` is an example; use a suitable available port.

### 6.3 Jupyter usage rules and port selection

Jupyter is a web-based platform for code development, analysis, and debugging. Examples of its setup and features follow in section 7.

For an ordinary batch job, you can leave SSH after `sbatch` submission and return later for results. **For Jupyter, set up and maintain the SSH tunnel while accessing the browser session.** The source says to exit the session only after shutting down Jupyter from the browser or cancelling the job manually.

The default Jupyter port is **8888**. Change to another port if that port is already in use by another TC1 user.

The source gives this head-node command for checking a desired port:

```bash
netstat -tp | grep <port_number>
```

The example screenshot searches `888` and shows established/time-wait connections to compute nodes on port `8885`, plus a notice that non-owned process details require root to see them all.

The guide interprets no matching output as the port being free. Preserve the intended check, but note its limitation: this output alone does not comprehensively prove that a port is unused on the target compute node. Use the address and port actually reported by the running Jupyter job.

**For your own applications, install packages in your own Conda environment via the command line. When the code is ready for computation, execute it directly by submitting a job script, rather than launching the finished computation through Jupyter.**

## 7. Miscellaneous: Jupyter, SSH tunnelling, and monitoring

*Original pages 24–29.*

### 7.1 What SSH tunnelling does

The cluster is accessed through SSH. To use JupyterLab or Notebook in a browser, set up an **SSH tunnel**, also called **SSH port forwarding**.

The tunnel is an encrypted SSH connection between a client and server through which a service port is relayed.

The source's example workflow is:

1. Use PuTTY on Windows to access TC1 and prepare your own Conda environment for Jupyter.
2. Submit a SLURM script to launch Jupyter and read its connection details from the job's error log.
3. Configure the SSH tunnel in PuTTY.
4. Open the local forwarded URL in your browser.

### 7.2 Step 1: create your own Jupyter environment

After connecting to the head node, execute:

```bash
module load anaconda
conda create -n RunJupyter
conda activate RunJupyter
conda install -c conda-forge jupyterlab notebook
conda install -c conda-forge ipykernel
conda deactivate
```

Source references:

- [Jupyter installation](https://jupyter.org/install)
- [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
- [Jupyter Notebook tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

You can alternatively use the shared Jupyter environment from section 6.

### 7.3 Step 2: create and submit the Jupyter job

The source names its script `run1.sh`. Its script template is:

```bash
#!/bin/sh
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --job-name=run1
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
source activate RunJupyter
# Add ONE of the Jupyter launch commands below here.
```

For **Notebook**, append this command, replacing `XXXX` with the desired port:

```bash
jupyter-notebook --ip=$(hostname -i) --port=XXXX
```

For **JupyterLab**, append this instead:

```bash
jupyter-lab --ip=$(hostname -i) --port=XXXX
```

Submit the completed script from the head-node command line:

```bash
sbatch run1.sh
```

As with all source examples, use the partition/QoS assigned to you and keep resource/time requests within your actual limits. Jupyter itself runs in the allocated compute-node job.

### 7.4 Step 3: read the connection information from the error log

The source creates the error log in the directory where the job was submitted, named:

```text
error_run1_<jobid>.err
```

Its example uses `xxxx` as the job-ID placeholder. Read the end of the log:

```bash
tail error_run1_<jobid>.err
```

Reference: [Linux tail examples](https://www.geeksforgeeks.org/tail-command-linux-examples/).

Record all three items from the log:

- The **compute-node IP address**.
- The **port number** actually used by Jupyter.
- The **browser URL and token**.

The screenshots show Jupyter Server `2.14.2` and two example service URLs:

```text
http://10.128.10.11:8883/lab?token=<token_from_your_log>
http://127.0.0.1:8883/lab?token=<token_from_your_log>
```

They also show a Jupyter runtime HTML file under the example account's `.local/share/jupyter/runtime/` directory and the server message that **Ctrl+C** stops that server and shuts down its kernels. The guide's normal shutdown/cancellation procedure is given below.

**Use your own job's details.** `10.128.10.11` is an example compute-node address, not the head-node SSH address. The token is generated for the running Jupyter server; copy it from your log rather than using an old screenshot value.

### 7.5 Steps 4–5: create the tunnel in PuTTY

1. Right-click the **top bar** of your current PuTTY SSH session.
2. Select **Change Settings…**.
3. Expand **SSH** and select **Tunnels** in the category tree, under the connection settings.
4. From your log's compute-node URL, identify the destination IP and port.
5. Enter the forwarding fields. The source's example is:

   ```text
   Source port:  8883
   Destination:  10.128.10.11:8883
   Type:         Local
   Address mode: Auto
   ```

6. Click **Add**. The entry should appear in **Forwarded ports**.
7. Click **Apply** to apply the change.

The source refers to the first URL in one passage as `http://10.128.10.11:8883/tree...`, while its JupyterLab screenshot uses `/lab?token=...`. The forwarding destination is the compute-node **IP and port**; use the actual application path and token reported by your job for browser access.

### 7.6 Step 6: open the locally forwarded browser URL

After the tunnel is active, paste the second, local URL into your local machine's browser:

```text
http://127.0.0.1:8883/lab?token=<token_from_your_log>
```

Keep the token from the error log and use the matching forwarded local port.

On successful connection, Jupyter displays an explorer/file-browser view of the directory from which you submitted the job script. **That directory becomes the `/` root shown inside Jupyter's file browser.** This is Jupyter's file-browser root, not a new authorisation to browse the operating system's root or other users' files.

The screenshot shows **Notebook**, **Console**, and **Other** launcher sections, with entries such as Python kernels, **Terminal**, **Text File**, **Markdown File**, **Python File**, and contextual help.

### 7.7 Open a terminal and activate other environments

Select **Terminal** under **Other**. Use it to execute commands on the allocated compute node and activate your own environments for testing:

```bash
conda env list
conda activate <environment_name>
```

The screenshot also demonstrates `nvidia-smi` from this compute-node terminal, showing the assigned Tesla V100-PCIE-32GB and no running GPU processes at that moment. This is the compute-node verification workflow described in section 3, not a head-node GPU check.

### 7.8 Upload files from Jupyter

Click **Upload Files** to choose files from your local drive and upload them to the root path `/` shown in Jupyter. As above, this corresponds to the job's Jupyter starting directory.

### 7.9 View PDF user guides in Jupyter

The source suggests installing the extension **`jupyterlab-pdf-preview`** to view PDF user guides available in TC1:

1. Open **Extension Manager**.
2. Find and install `jupyterlab-pdf-preview` in your own setup.
3. Copy the required guide from the authorised shared guide directory to your Jupyter starting directory:

   ```text
   /tc1share2/0_TC1_Guides
   ```

4. Right-click the PDF file to open it for viewing.

The screenshot's copy example is:

```bash
cp /tc1share2/0_TC1_Guides/CCDSGPU-TC1-UG-UserGuide.pdf .
```

The shared-directory screenshot also shows resource-request forms, FYP/project forms, a GPU guide, a VSCode user guide, job-script samples, VPN material, a student video guide, a `README`, and a notice that the folder contains forms/guides for download. Consult the files actually present in that shared directory.

The PDF-preview screenshot opens `CCDSGPU-TC1-VSCode_UserGuide.pdf`. The visible first page of that separate guide begins with **Install OpenSSH Client in Windows Desktop** and instructs Windows users to access **System → Optional features**, search for **openssh**, and select **OpenSSH Client** for installation. Its visible reference is [Microsoft OpenSSH installation and first use](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui). This is an example of reading another guide through the PDF viewer; it is not an extra prerequisite imposed by the main TC1 guide.

### 7.10 Monitor GPU usage with nvtop

From the Jupyter compute-node session, launch **another terminal** and execute:

```bash
nvtop
```

Monitor the GPU and GPU-memory usage of your process. Press **Q** or **F10** to exit.

Reference: [nvtop GPU monitoring](https://www.cyberciti.biz/hardware/nvtop-command-in-linux-to-monitor-nvidia-amd-intel-gpus/).

### 7.11 Other compute-node monitoring commands

The source lists:

```bash
# Summarised memory and swap information in human-readable units
free -h

# CPU statistics
mpstat

# Real-time view of running processes, omitting idle tasks
top -i

# Process listing example as printed in the source
ps -u
```

The source describes `ps -u` as displaying your own processes, but it does not supply a user argument. To make the intended user selection explicit, use the following form with your own account ID:

```bash
ps -u <username>
```

These monitoring examples occur within the allocated compute-node/Jupyter workflow.

### 7.12 Shut down Jupyter and release resources

The source provides two ways to stop the service:

1. In Jupyter, open **File → Shut Down**.
2. Alternatively, cancel the SLURM job from a terminal:

   ```bash
   squeue -u <username>
   scancel <jupyter_jobid>
   ```

The screenshot cancels example job `127`. After cancellation, the browser can show **Server Connection Error**, stating that it cannot connect to the Jupyter server and will continue trying to reconnect. This is expected after deliberately terminating the server/job.

**Cancel your job when you no longer need it.** This releases its allocated CPU, memory, and GPU back to the pool for others. Maintain the SSH tunnel while using Jupyter, and finish/shut down the job before leaving the session as instructed by the source.

### 7.13 macOS and Linux tunnel access

The source does **not** give an SSH command-line forwarding example. It advises macOS/Linux users to look up the appropriate SSH tunnelling command or use another SSH client to facilitate the connection.

It repeats these client references:

- [PuTTY/SSH on Mac](https://www.ssh.com/academy/ssh/putty/mac)
- [Termius for macOS](https://termius.com/free-ssh-client-for-mac-os)

Use the same compute-node destination, service port, and local URL principles described above when configuring a different client.

## 8. Applying for additional resources

*Original page 30.*

### 8.1 Additional QoS

You may request an additional QoS for:

- Longer computation times: **8 hours, 12 hours, 24 hours, up to a maximum of 48 hours**.
- More CPU, memory, and/or GPU-card resources.

**Usage period:** no more than **one month**.

Application process:

1. Provide a valid reason for the request.
2. Provide verification: your **latest code and computation results from TC1 using your existing assigned QoS**, demonstrating that the current resources cannot support your computation requirements.
3. Email **CCDSGPU-TC-Support** at [CCDSgpu-tc@ntu.edu.sg](mailto:CCDSgpu-tc@ntu.edu.sg).
4. The administrator evaluates the purpose and actual requirements, then assigns an appropriate QoS if approved.
5. You will be notified if the application succeeds.

### 8.2 Additional storage

The source permits applying for additional computation-data storage, described as **more than 100 GB**. A shared folder is created on another storage system and assigned to the user.

- **Usage period:** four months.
- **Maximum quota:** depends on available free space and approval.

Application process:

1. Email **CCDSGPU-TC-Support** at [CCDSgpu-tc@ntu.edu.sg](mailto:CCDSgpu-tc@ntu.edu.sg).
2. Provide the reason and verification for the request.
3. Complete the application form sent to you by email. The source calls it a “PFD form”; this appears to mean a PDF form.
4. You will be notified of the resource assignment if the application succeeds.

### 8.3 Terms and conditions for additional resources

- Approval depends on usage purpose, resource availability, cluster workload, and administrative criteria.
- Usage of assigned resources is monitored.
- The administrator may revoke resources if they are underused or adversely affect cluster operation.
- **Assigned resources are removed when their usage period expires, without prior notification.**

## 9. Important notices and support

*Original page 31.*

1. **General guide only.** The guide is intended to get you started. Use its reference links or search for more detailed information when required.

2. **Academic use only.** GPU resources may be used only for the purpose for which they were granted. Using them for another purpose can result in account termination. Authorised personnel may intercept, monitor, record, and inspect activities in TC1. Unauthorised or improper use may result in administrative/disciplinary action and civil or criminal consequences.

3. **No unauthorised directory access.** Do not access directories other than your assigned home directory and authorised shared folders. Unauthorised access to other files or directories, **even just for browsing**, is treated as a security breach. Serious action may be taken against users who breach the cluster's security.

4. **Access and computation are monitored.** Computation that disrupts cluster operation, and jobs using an unassigned QoS, can lead to termination of access and computation **without prior notification**. Severe offenders can be barred from the service.

5. **No data backup service.** You are responsible for protecting and maintaining your own data.

6. **Coursework/project questions:** consult your **supervisor or Teaching Assistant (TA)**.

7. **Technical issues:** email **CCDSGPU-TC-Support** at [CCDSgpu-tc@ntu.edu.sg](mailto:CCDSgpu-tc@ntu.edu.sg).

## Conversion notes and source inconsistencies

This section records clarifications made while converting the PDF so that source quirks do not become silent, misleading instructions.

1. **Scope and formatting:** all nine substantive source sections are included, together with the cover's title, classification, source date, section/page mapping, and information recovered from diagrams and screenshots. Repeated headers/footers, decorative formatting, window chrome, and repeated command prompts are not duplicated. Screenshot account names, IDs, package versions, and utilisation values are examples, not user-specific assignments or current inventory. Example Jupyter token strings are represented by `<token_from_your_log>` so readers use their own server's token.

2. **Partition names:** most updated scripts and screenshots use `UGGPU-TC1`. Page 14 calls the partition `CCDSGPU_UG`, and page 17's intentionally malformed-script screenshot uses that older label. This document retains the discrepancy and uses `UGGPU-TC1` in the main script examples. Check `scontrol show partition` and `MyTCinfo` for the partition you may actually use.

3. **CPU option spelling:** page 17 prints `--cpus-per-tasks=2`. The script reference above uses the corrected singular spelling `--cpus-per-task=2`; the original spelling is explicitly recorded in section 5.5.

4. **Node-count typography:** the table on page 18 shows `*--nodes=1`, mixing its required-field asterisk into the displayed option. The actual directive is `#SBATCH --nodes=1`, as in the source's complete scripts.

5. **Conda removal typography:** page 10 uses a typographic dash in `conda env remove –name ...`, while its screenshot illustrates deleting a full environment path. The usable named form here is `conda env remove --name <environment_name>`. The source's ambiguous `~<your home directory>/.conda/envs/<folder>` is expressed as `~/.conda/envs/TestEnv` with the ownership/deactivation conditions retained.

6. **Shell initialization path:** `source .bashrc` in the source assumes the current directory is home. This conversion uses `source ~/.bashrc` for the same home-directory file.

7. **Environment YAML update:** page 9 describes `conda env update --prefix ./TestEnv --file TestEnv.yml --prune` as updating the content of the `.yml` file. The operation applies the file's configuration to the target environment; the command itself is preserved.

8. **Environment renaming:** page 9 says there is no rename command and gives clone-then-remove instructions. The clone/remove workflow is preserved here without making a claim about every Conda version's available commands.

9. **QoS time/job descriptions:** page 14 explains `MaxJobsPU` as concurrent running jobs and gives `MaxWall=06:00:00`; page 16 separately uses `MaxSubmit=2` and says total running computation must stay within six hours. Page 20 checks each job's time against `MaxWall` and says pending jobs expire after a wait referenced to that limit. These statements are retained with their source context. Inspect your live assignment and ask support if their application is unclear.

10. **Pending is a state, not necessarily a failed computation:** the source sometimes describes a `PD` job as having “failed to run.” This conversion explains it as pending and retains the source's reasons and remedies. A job waiting for resources may run later.

11. **Output-filename case:** page 21's prose capitalises example names as `Output_MyJob_xxxx.out` and `Error_MyJob_xxxx.err`, although the directives and screenshot use lowercase `output_` and `error_`. This conversion follows the actual lowercase directives. Filenames are case-sensitive.

12. **Jupyter paths and ports:** `/tree` appears in prose while `/lab` appears in the Lab screenshots. Several demonstrations use different ports (`8881`, `8882`, `8883`, `8885`, and the default `8888`). They are distinct examples, not a sequence requiring those exact values. Use the running job's reported address, actual port, path, and token.

13. **Port-check limitation:** the original says no match from `netstat -tp | grep <port>` means the port is free. The check is preserved, with a clarification that this head-node connection listing alone does not prove availability on the target compute node.

14. **Process-list syntax:** page 28 prints `ps -u` and describes it as showing your own processes. The original example is preserved and the explicit user-selection form `ps -u <username>` is provided.

15. **GPU-memory and system-memory requests:** the source's 32 GB V100 memory is a fixed hardware property, separate from the job's `--mem` system-memory request. GPU-card index `IDX:0` identifies the first card; `JOB_GRES=gpu:1` specifies a count of one GPU.

16. **Application-form typo:** page 30 says “PFD form.” The application process is retained and the likely intended “PDF form” is identified, rather than silently changing the source wording.

17. **Reference links:** all distinct online reference URLs in the PDF text are retained in the appropriate sections, including dated or non-HTTPS links. They are source references and were not independently checked or updated during this conversion.

18. **No live cluster validation:** the commands and resource values were converted from the PDF and its screenshots. No SSH login, GPU job, package installation, resource application, or support email was performed as part of this document conversion.

*End of consolidated guide.*
