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

## Replace `(...)` with `[...]`

```bash
find [DIR] -type f -name "*.json" -exec sed -i 's/(/[/g; s/)/]/g' {} \;
```

## Run program

```bash
python main.py [DIR_NAME] [BATCH_SIZE]
```
