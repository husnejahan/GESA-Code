# """

# Author: Most Husne Jahan
# December -2023

# Generate PCN-Attack Dataset

# Create env:

# python3 -m venv my-env
# source my-env/bin/activate
# pip3 install -r requirements.txt
# pip3 install open3d

# sh PCDtoPCD.sh
#  sh NPYtoPCD.sh

# """


# python3 PCN_PCDDown.py --mode Train
# python3 PCN_PCDDown.py --mode Val
# python3 PCN_PCDDown.py --mode Test



python3 PCN_PCDDownAll.py --mode Train
python3 PCN_PCDDownAll.py --mode Val
python3 PCN_PCDDownAll.py --mode Test