# Dataset split files

The training entry point expects `train.txt` and `val.txt` in this directory.
Each line must contain two paths relative to the training dataset root:

```text
images/example.png masks/example.png
```

The split files used on the private experiment server are intentionally not
included in this public repository.
