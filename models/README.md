Models are not version controlled directly, but rather using
[DVC](https://dvc.org/) for better reproducibility.

## How to validate downloaded models
DVC stores a hash of the files or directories it tracks, so you can
be sure your version of the model matches the one over which
experiments were executed.

We use `dvc status` to check it, and the following output is the
expected one:

```shell
$ dvc status
Data and pipelines are up to date
```

## How to download models

TODO: Add models to DVC registry

### NATS-Bench
Follow Step 2 of the
["Preparation and Download" instructions](https://github.com/D-X-Y/NATS-Bench?tab=readme-ov-file#preparation-and-download)
provided alongside the official benchmark API implementation by
its authors.
Each tarball takes up approximately 1 GB when unpacked.
As of writing, the files were available at
[this](https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt)
Google Drive folder.
