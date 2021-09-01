curl -O https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb
dpkg -i packages-microsoft-prod.deb
#apt-get update
apt-get install blobfuse
mkdir -p /mnt/ramdisk
mount -t tmpfs -o size=1400g tmpfs /mnt/ramdisk
mkdir -p /mnt/ramdisk/blobfusetmp
mkdir -p /home/leg/deepbiosphere/data/NAIP
chown -R leg: /mnt/ramdisk
chown -R leg: /home/leg/deepbiosphere/data
blobfuse /home/leg/deepbiosphere/data/NAIP --tmp-path=/mnt/ramdisk/blobfusetmp  --config-file=/home/leg/deepbiosphere/scripts/azure/fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o allow_other -o negative_timeout=120 -o nonempty
