from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Kontext-dev",
    cache_dir="/ntuzfs/shilin/.cache/huggingface/hub",
)

print("仓库已下载到：", local_dir)
