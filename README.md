# DiscoverMotorPrograms
Cleaned up codebase for motor programs.

Run the following commands after sourcing appropriate environment and navigating to one folder above DiscoverMotorPrograms repository: 

### To run Skill Net pre-training with pre-generated plan data: 

python -m DiscoverMotorPrograms.Experiments.abstraction.mime_plan_skill --plot_scalars --display_visuals=True --display_freq=5000 --name=M1_NewMIMESkillNet --st_space=joint_both --MIME_dir=/checkpoint/tanmayshankar/MIME/ --num_epochs=100 --lr_step_epoch_freq=50 --optim_bs=2 --variable_ns=False --gpu_id=0 --align_loss_wt=1.0 --nz=64 --nh=64 --pred_gripper=True

### To run Fixed Number of Segments model with (fixed) pre-trained skill net:

python3 -u -m DiscoverMotorPrograms.Experiments.abstraction.mime --display_freq=5000 -optim_bs=2 --variable_ns=False --st_space=joint_both_gripper --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --novariable_nseg --num_epochs=300 --lr_step_epoch_freq=50 --name=M2_NewMIMEFNSeg --lpred_p_bias=0 --len_loss_wt=0.01 --step_loss_wt=2. --align_loss_wt=0.1 --transformer --nonormalize_loss --checkpoint_dir=saved_models/ --pretrain_skillnet_name=M1_NewMIMESkillNet

### To run Fixed Number of Segments model while finetuning the skill net: 

python3 -u -m DiscoverMotorPrograms.Experiments.abstraction.mime --display_freq=5000 -optim_bs=2 --variable_ns=False --st_space=joint_both_gripper --num_pretrain_skillnet_epoch=100 --fixed_skillnet=False --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --novariable_nseg --num_epochs=300 --lr_step_epoch_freq=50 --name=M3_NewMIMEFNSeg_finetune --lpred_p_bias=0 --len_loss_wt=0.01 --step_loss_wt=2. --align_loss_wt=0.1 --transformer --nonormalize_loss --checkpoint_dir=saved_models/ --pretrain_skillnet_name=M1_NewMIMESkillNet --num_pretrain_epochs=100 --network_dir=saved_models/M2_NewMIMEFNSeg

### To run Variable Number of Segments with fixed pre-trained skill net: 

python3 -u -m DiscoverMotorPrograms.Experiments.abstraction.mime --display_freq=5000 -optim_bs=2 --variable_ns=False --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --variable_nseg --num_epochs=300 --lr_step_epoch_freq=50 --name=M4_NewMIMEVNSeg --lpred_p_bias=0 --len_loss_wt=0.01 --step_loss_wt=2. --align_loss_wt=0.1 --transformer --nonormalize_loss --checkpoint_dir=saved_models/ --pretrain_skillnet_name=M1_NewMIMESkillNet --lpred_p_bias=0.5 --st_space=joint_both_gripper

### To run Variable Number of Segments while finetuning pre-trained skill net: 

python3 -u -m DiscoverMotorPrograms.Experiments.abstraction.mime --display_freq=5000 -optim_bs=2 --variable_ns=False --num_pretrain_skillnet_epoch=100 --fixed_skillnet --nz=64 --nh=64 --MIME_dir=/checkpoint/tanmayshankar/MIME/ --variable_nseg --num_epochs=300 --lr_step_epoch_freq=50 --name=M5_NewMIMEVNSeg_finetune --lpred_p_bias=0 --len_loss_wt=0.01 --step_loss_wt=2. --align_loss_wt=0.1 --transformer --nonormalize_loss --checkpoint_dir=saved_models/ --pretrain_skillnet_name=M1_NewMIMESkillNet --lpred_p_bias=0.5 --st_space=joint_both_gripper --num_pretrain_epochs=100 --network_dir=saved_models/M3_NewMIMEVNSeg

