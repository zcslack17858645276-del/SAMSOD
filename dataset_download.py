import os
import oss2
import json
import zipfile

# generate the file manifest
def generate_oss_manifest(
    endpoint,
    bucket_name,
    access_key_id,
    access_key_secret,
    oss_prefix,
    output_json
):
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    files = []

    for obj in oss2.ObjectIterator(bucket, prefix=oss_prefix):
        if obj.key.endswith("/"):
            continue

        files.append({
            "path": obj.key[len(oss_prefix):],
            "oss_key": obj.key,
            "size": obj.size
        })

    manifest = {
        "dataset": oss_prefix.strip("/").split("/")[-1],
        "base_prefix": oss_prefix,
        "files": files
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✅ Manifest written to {output_json}")
    print(f"Total files: {len(files)}")

# single file download
def download_oss_dir(
    endpoint,
    bucket_name,
    access_key_id,
    access_key_secret,
    oss_prefix,
    local_dir
):
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    os.makedirs(local_dir, exist_ok=True)

    for obj in oss2.ObjectIterator(bucket, prefix=oss_prefix):
        if obj.key.endswith("/"):
            continue

        local_path = os.path.join(
            local_dir,
            obj.key[len(oss_prefix):].lstrip("/")
        )

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {obj.key} -> {local_path}")
        bucket.get_object_to_file(obj.key, local_path)

    print("✅ Download finished.")

# multiple file download
def download_from_manifest(
    manifest_json,
    endpoint,
    bucket_name,
    access_key_id,
    access_key_secret,
    output_dir
):
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    for file in manifest["files"]:
        local_path = os.path.join(output_dir, file["path"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path):
            continue

        print(f"Downloading {file['oss_key']} -> {local_path}")
        bucket.get_object_to_file(file["oss_key"], local_path)

    print("✅ Download completed.")

# download other file(some different, only for datasets)
def download_datasets_from_manifest(
    manifest_json,
    endpoint,
    bucket_name,
    access_key_id,
    access_key_secret,
    output_dir,
    anonymous=False
):
    """
    根据 manifest JSON 下载 zip 文件

    manifest_json: 你给的那个 JSON
    output_dir:    zip 保存目录
    anonymous:     True = bucket 公共读，不用 AK
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if anonymous:
        auth = oss2.AnonymousAuth()
    else:
        auth = oss2.Auth(access_key_id, access_key_secret)

    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    for item in manifest["files"]:
        zip_name = os.path.basename(item["path"])
        oss_key = item["oss_key"]
        local_zip = os.path.join(output_dir, zip_name)

        if os.path.exists(local_zip):
            print(f"Skip existing: {zip_name}")
            continue

        print(f"Downloading {oss_key} -> {local_zip}")
        bucket.get_object_to_file(oss_key, local_zip)

    print("✅ Zip download finished.")

# unzip the zip file
def unzip_all(
    zip_dir,
    extract_to,
    remove_zip=False
):
    """
    解压 zip_dir 下所有 zip 文件

    remove_zip=True 可在解压后删除 zip
    """

    os.makedirs(extract_to, exist_ok=True)

    for fname in os.listdir(zip_dir):
        if not fname.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, fname)
        print(f"Extracting {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

        if remove_zip:
            os.remove(zip_path)

    print("✅ All zip files extracted.")

if __name__ == "__main__":
    # 经过测试，以下功能（获取bucket中文件数据、下载、解压等）均可完成，处于个人完全考虑，将会暂时关闭ACCESS，若需要测试，请联系3331238758@qq.com
    ENDPOINT="https://oss-cn-beijing.aliyuncs.com"
    BUCKET_NAME="data-sam2"
    ACCESS_KEY_ID=""
    ACCESS_KEY_SECRET=""
    # generate the json of datasets
    generate_oss_manifest(
        endpoint=ENDPOINT,
        bucket_name=BUCKET_NAME,
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        oss_prefix="datasets",
        output_json="oss_json/datasets_manifest.json"
    )

    # generate the json of checkpoints
    generate_oss_manifest(
        endpoint=ENDPOINT,
        bucket_name=BUCKET_NAME,
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        oss_prefix="checkpoints/",
        output_json="oss_json/checkpoints_manifest.json"
    )

    '''
    '''
    # download datasets
    download_datasets_from_manifest(
        manifest_json="oss_json/datasets_manifest.json",
        endpoint=ENDPOINT, # oss-cn-beijing.aliyuncs.com
        bucket_name=BUCKET_NAME, #data-sam2
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        output_dir="dataset"
    )
    '''
    '''
    # unzip datasets
    unzip_all(
        zip_dir="dataset",
        extract_to="dataset",
        remove_zip=False
    )

    # download checkpoints
    download_from_manifest(
        manifest_json="oss_json/checkpoints_manifest.json",
        endpoint=ENDPOINT,
        bucket_name=BUCKET_NAME,
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        output_dir="checkpoints"
    )
    

