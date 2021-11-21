# CE7454_FishNet_Project
This code utilizes the original FishNet model published here https://github.com/kevin-ssy/FishNet 
as well as the DeepFish Dataset, whose code is published here https://github.com/alzayats/DeepFish

This project applies the FishNet Model on the DeepFish dataset and shows that on the counting and classification tasks, we can achieve better performance than that of the ResNet50 model reported in the DeepFish paper. 

The DeepFish code has been cloned and reworked quite extensively for this project, the code from the original source will not run as intended here.  

The model classes in this repository will load the FishNet150 model from the original FishNet code as a module.
It will also make a reference to kevin-ssy's pre-trained FishNet150 model, and requires the checkpoint file for the model pre-trained without tricks 
https://www.dropbox.com/s/hjadcef18ln3o2v/fishnet150_ckpt.tar?dl=0 
If hyperlink doesa not work refer back to the author's original github


