# Pics

## Set up environment

```bash
conda env create -f environment.yaml
# conda init bash
# source ~/.bashrc
```

## Activate environment

```bash
conda activate graphenv
```

## Reinstall after updating environment

```bash
conda deactivate
conda env update --file environment.yaml --prune
```

## Run program

```bash
python plot_graphs.py [DIR_NAME]
```
