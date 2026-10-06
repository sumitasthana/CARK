# Keep experiment files in Cloudflare R2

R2 keeps files when a Colab session ends. A bucket is a named place to keep those files. The notebook downloads the chosen model to Colab, trains there, and uploads the report after each pass through the training images. When training finishes, it uploads and checks the final model before marking the lesson complete in R2.

This changes where files are saved. The next experiment still starts from your model after tasks 3, 0, 9, and 5, and teaches only task 17 with beta 0.01.

## Set up storage once

1. Open Cloudflare, then **R2 object storage**. If R2 is not enabled, review its billing terms before enabling it. A Cloudflare account alone does not enable R2. See [Cloudflare's R2 setup instructions](https://developers.cloudflare.com/r2/get-started/).
2. Create a bucket named `uncle-experiments`, or use another name and change `BUCKET` in the notebook. Keep the bucket private. Use Standard storage. See [creating a bucket](https://developers.cloudflare.com/r2/buckets/create-buckets/).
3. In the R2 overview, open **Manage API Tokens**. Create an R2 token with **Object Read & Write**, limited to this bucket. Save its **Access Key ID** and **Secret Access Key**. These are the two values the Python client needs. A general Cloudflare API token is different. See [R2 credentials](https://developers.cloudflare.com/r2/api/tokens/).
4. Find the bucket's **S3 API endpoint**. Use the server address, such as `https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com`, without a trailing slash or bucket name. A bucket with a jurisdiction setting has a different server address; keep that jurisdiction in the address shown by Cloudflare.
5. Open notebook 15 in Colab. Click the key icon in the left sidebar to open **Secrets**. Add `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY`, using the values above. Turn on notebook access for each. Keep credentials out of notebook cells, Git, and chat.

R2 charges for storage and some requests. Downloads have no egress charge. Check [current pricing](https://developers.cloudflare.com/r2/pricing/) before copying a large collection. Verification reads every uploaded file back, which also uses read requests.

## Copy existing files on CPU

Open [notebook 15](https://colab.research.google.com/github/sumitasthana/CARK/blob/main/notebooks/15_R2_learning_loss.ipynb). Choose a CPU runtime. Leave `MODE = "COPY"` and run the cells in order.

The notebook copies all files under `/content/drive/MyDrive/uncle` to the same folder names in R2. It reads each copied file back and compares its SHA256, a fingerprint calculated from the file's bytes. Drive originals stay in place. A second attempt checks existing copies and continues; a different file at the same R2 name stops the copy.

Close other notebooks that are writing to this Drive folder before copying. Use one session to write to each R2 experiment folder. Checkpoint overwrite protection checks before uploading; it is not a lock between simultaneous sessions.

Only files under that Drive folder are copied. Code stays in Git. Tiny ImageNet is downloaded separately by the training notebook, as before. If another project folder also needs copying, change `DRIVE_ROOT` and `COPY_PREFIX` together and keep the model's `SOURCE_KEY` pointed at its copied location.

## Run the next lesson

After the CPU copy prints that files were checked, open a fresh A100 runtime. Set `MODE = "RUN"` in notebook 15 and run downward. Drive is only mounted in COPY mode.

The default model is your confirmed file:

`uncle/learning_initialization/20261006_hyperfan_L3_L0_L9_seed0_02/checkpoint_seq1_resnet50_seed0.pt`

The notebook checks that this file has learned 3, 0, 9, and 5 using Hyperfan-in. It prints its saved scores before training. The expected scores are 27.0%, 31.2%, 48.4%, and 54.8%. Teaching starts only after these checks.

New results go to `uncle/learning_loss_diagnostic/20261006_learning_loss_r2_01`. Each beta starts from the same source model. Reports are uploaded after each pass. The final model is uploaded at the end of the lesson, so an interrupted lesson has reports but cannot resume midway through training. Use a new experiment name for that retry. Earlier tasks do not need to be trained again.

The final cell checks the saved model and the copies in R2, then releases the GPU. A session that dies after an upload leaves its uploaded files in R2. In a later session, `store.keys("uncle/learning_loss_diagnostic/")` lists results, and `store.download(key, CACHE)` gets and checks a file.

The transfer code uses Cloudflare's [supported boto3 connection](https://developers.cloudflare.com/r2/examples/aws/boto3/). Automated tests check file corruption, retries, cache replacement, and publishing order using a local fake service. A live connection must be checked with your bucket and credentials.
