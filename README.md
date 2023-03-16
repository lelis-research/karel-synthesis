# Karel Synthesis

This work-in-progress repository implements a symbolic DSL for Karel and implements the Program VAE method from [LEAPS](https://clvrai.github.io/leaps/).

## Instalation

Download the LEAPS dataset from [this link](https://drive.google.com/drive/folders/1CM4_1zBAXgztPX6n_D6HmavYujZSfdV4), and extract on `data/program_dataset`. The file tree should look like this:

```
data
└── program_dataset
    ├── data.hdf5
    └── id.txt
```

Install the requirements from PyPI:

```bash
pip install -r requirements.txt
```

## Program VAE implementation

The Program VAE models are implemented in `vae/models`. `base_vae.BaseVAE` contains the base implementation of common functions and modules for Program VAEs and `leaps_vae.LeapsVAE` implements the Program VAE from the LEAPS paper.

## Program dataset pre-loading

In order to speed up the trainer, the script `main_dataset.py` can be used to pre-load the dataset to a pickle file:

```bash
python3 main_dataset.py
```

## Training Program VAE

LeapsVAE can be trained by running `main_trainer.py`:

```bash
python3 main_trainer.py
```

This command instantiates a `Trainer` from `vae/trainer.py` and runs it directly.

For information about the available arguments, run the script command:

```bash
python3 main_trainer.py --help
```

## Latent Search

Also known as CEM in the LEAPS paper, Latent Search is implemented in `search/latent_search.py`. The script `main_latent_search.py` contains a wrapper to that class and can be called directly to execute Latent Search. In order to get the same behaviour as the original LEAPS paper, execute it with the following arguments:

```bash
python3 main_latent_search.py --env_enable_leaps_behaviour --search_reduce_to_mean
```

The available arguments are the same as the trainer script, and can similarly be visualized through:

```bash
python3 main_latent_search.py --help
```