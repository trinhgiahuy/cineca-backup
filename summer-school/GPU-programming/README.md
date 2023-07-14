# GPU programming


## Slides

The slides are available in PDF format
```bash
cd slides
```

## Exercises

A skeleton of exercises are provided

#### OpenACC exercises

```bash
cd hands-on/openacc
```

#### CUDA exercises

```bash
cd hands-on/cuda
```

At the end of course the solutions of all exercises are published in the repository

## Load Environment

#### PGI environment

```bash
module load profile/advanced
module load pgi
module load cuda
```

#### GNU environment

```bash
module load profile/advanced
module load gnu
module load cuda
```

#### Show loaded environment
```bash
module list
```

### Using GPU reservation
```bash
srun -N1 -n4 -A train_scB2018 --reservation=s_tra_schoolBgpu --gres=gpu:1 -t 20:00 -p gll_usr_gpuprod --pty bash
```

