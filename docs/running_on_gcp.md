# Running on Google Cloud

The intended workflow: build and test on the small data in Google Colab, then run the same notebook on a Dataproc cluster with the big data. The notebook works out where it's running and handles the setup for each environment.

## 1. Put the data in Cloud Storage

In your bucket, create a folder called `data_for_project` and upload:

```
your-bucket-name/
  data_for_project/
    small_dataset.parquet          single file, 9.6 MB
    big_dataset.parquet/           upload as a folder: part-00 ... part-15
```

Upload `big_dataset.parquet` with **Upload folder**, not as loose files. Spark reads the folder as one table, so don't rename the part files.

## 2. Configure the notebook

In the config cell (third cell):

```python
project_id = 'your-gcp-project-id'     # the project ID, not the project name
bucket = 'your-bucket-name'
path_to_data_files = "/data_for_project/"
data_file_name = "small_dataset.parquet"
```

## 3. Small data in Colab

1. Open [colab.research.google.com](https://colab.research.google.com), then **File > Upload notebook**.
2. **Runtime > Run all**. Sign in when the Google auth prompt appears.
3. Takes 15 to 20 minutes. It's done when the evaluation cell prints `MODEL EVALUATION RESULTS (small_dataset.parquet)` and the last cell shows ten recommendations for one listener.

Don't run the big data in Colab. It runs out of memory, which is the reason for Dataproc.

## 4. Dataproc cluster

The master node needs at least 16 GB of RAM. An 8 GB master ran out of memory during model training. Only the Jupyter component is needed.

```bash
gcloud dataproc clusters create recommender-cluster \
    --region=us-east4 \
    --num-masters=1 \
    --master-machine-type=n4-highmem-2 \
    --master-boot-disk-size=50GB \
    --num-workers=2 \
    --worker-machine-type=n4-standard-2 \
    --worker-boot-disk-size=50GB \
    --optional-components=JUPYTER \
    --max-idle=7200s \
    --enable-component-gateway
```

Use the region your bucket is in. `--max-idle` deletes the cluster after 2 idle hours as a safety net. That's about $0.37 an hour for this setup at the time of writing.

## 5. Run on the cluster

1. Open the cluster, then **Web Interfaces > JupyterLab**.
2. Upload the notebook and choose the **PySpark** kernel.
3. With `data_file_name = "small_dataset.parquet"`, run **Kernel > Restart Kernel and Run All Cells**. The first cells should print `Not running on CoLab` and `Using the Spark session provided by Dataproc`. The results should match Colab.
4. Change one line, then restart and run all again:

   ```python
   data_file_name = "big_dataset.parquet"
   ```

5. Download the notebook with its outputs.

## 6. Delete the cluster

Dataproc bills for every minute the cluster is up, busy or idle.

```bash
gcloud dataproc clusters delete recommender-cluster --region=us-east4
```

## Troubleshooting

| You see | Fix |
|---|---|
| 403 Forbidden reading the bucket (Colab) | Re-run the auth cell and sign in with the account that owns the project. |
| `Downloaded 0 file(s)` or `FileNotFoundError` (Colab) | The file isn't at `path_to_data_files`. The read cell prints where it actually is in the bucket. |
| `Path does not exist` (Dataproc) | Same cause. `path_to_data_files` has to match the bucket folder exactly, slashes included. |
| `Unable to infer schema for Parquet` | `big_dataset.parquet` was flattened or is empty. Re-upload it as a folder. |
| `NameError: spark` (Dataproc) | Wrong kernel. Switch to PySpark and run all again. |
| `Py4JNetworkError: Answer from Java side is empty` | The driver ran out of memory. Use a bigger master (see step 4). |
| Quota error creating the cluster | Use fewer workers or smaller machines. 2 workers is enough for this data. |
