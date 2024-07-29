# """

# Author: Most Husne Jahan
# December -2023

# Generate GESA-PCN Dataset

# Create env:

# python3 -m venv my-env
# source my-env/bin/activate
# pip3 install -r requirements.txt
# pip3 install open3d

# sh PCDtoAttack.sh

# """


python3 GESA-PCN.py --mode Train
python3 GESA-PCN.py --mode Val
python3 GESA-PCN.py --mode Test

