# GESA: Exploring Loss-based Adversarial Attacks in Volumetric Media Streaming

# Overview


# Requirement

To reproduce the results of the paper, please set up the Python environment using the following code:

      git clone https://github.com/husnejahan/GESA-Code.git
      python3 -m venv my-env
      source my-env/bin/activate
      pip3 install -r requirements.txt
   
# GESA-PCN

To downsample in any resolution:

     cd GESA-PCN
     PCDtoPCD.sh

 To apply Gilbert-Elliott Shape Attack(GESA)  
 
     PCDtoAttack.sh

# GESA-Shape

To convert as .pcd file and downsample in any resolution:

     cd GESA-Shape
     NPYtoPCD.sh

To apply Gilbert-Elliott Shape Attack(GESA) 

     PCDtoAttack.sh


# Citation

Please cite this paper if you want to use it in your work.
