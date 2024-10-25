
sudo mount -t nfs4 -o nfsvers=4.1 fs-02402528f441d97bf.efs.us-east-1.amazonaws.com:/ /mnt/efs

cd /mnt/efs/DPDataset_public
sudo mount -o loop frames_squashed/2023-01-17-openoffice frames/2023-01-17-openoffice
sudo mount -o loop frames_squashed/2023-01-17-livingroom frames/2023-01-17-livingroom