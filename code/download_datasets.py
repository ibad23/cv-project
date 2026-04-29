import os
from huggingface_hub import snapshot_download

def download_qanastek():
    print("Downloading qanastek/FrenchLicencePlateDataset...")
    repo_id = "qanastek/FrenchLicencePlateDataset"
    local_dir = "datasets/qanastek_french_plates"
    os.makedirs(local_dir, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Dataset downloaded to {local_dir}")

if __name__ == "__main__":
    download_qanastek()
