from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from absl import flags, app
import pdb
import scipy.misc
import copy

import robosuite
from robosuite.wrappers import IKWrapper

class MujocoVisualizer():

    def __init__(self, has_display=False):

        # Create environment.
        print("Do I have a display?", has_display)
        self.base_env = robosuite.make('BaxterLift', has_renderer=has_display)
        # self.base_env = robosuite.make("BaxterViz",has_renderer=True)

        # Create kinematics object. 
        self.baxter_IK_object = IKWrapper(self.base_env)
        self.environment = self.baxter_IK_object.env        
    
    def update_state(self):
        # Updates all joint states
        self.full_state = self.environment._get_observation()

    def set_ee_pose_return_image(self, ee_pose, arm='right', seed=None):

        # Assumes EE pose is Position in the first three elements, and quaternion in last 4 elements. 
        self.update_state()

        if seed is None:
            # Set seed to current state.
            seed = self.full_state['joint_pos']

        if arm == 'right':
            joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
                target_position_right=ee_pose[:3],
                target_orientation_right=ee_pose[3:],
                target_position_left=self.full_state['left_eef_pos'],
                target_orientation_left=self.full_state['left_eef_quat'],
                rest_poses=seed
            )

        elif arm == 'left':
            joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
                target_position_right=self.full_state['right_eef_pos'],
                target_orientation_right=self.full_state['right_eef_quat'],
                target_position_left=ee_pose[:3],
                target_orientation_left=ee_pose[3:],
                rest_poses=seed
            )

        elif arm == 'both':
            joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
                target_position_right=ee_pose[:3],
                target_orientation_right=ee_pose[3:7],
                target_position_left=ee_pose[7:10],
                target_orientation_left=ee_pose[10:],
                rest_poses=seed
            )
        image = self.set_joint_pose_return_image(joint_positions, arm=arm, gripper=False)
        return image

    def set_joint_pose_return_image(self, joint_pose, arm='both', gripper=False):

        # FOR FULL 16 DOF STATE: ASSUMES JOINT_POSE IS <LEFT_JA, RIGHT_JA, LEFT_GRIPPER, RIGHT_GRIPPER>.

        self.update_state()
        self.state = copy.deepcopy(self.full_state['joint_pos'])
        # THE FIRST 7 JOINT ANGLES IN MUJOCO ARE THE RIGHT HAND. 
        # THE LAST 7 JOINT ANGLES IN MUJOCO ARE THE LEFT HAND. 
        
        if arm=='right':
            # Assume joint_pose is 8 DoF - 7 for the arm, and 1 for the gripper.
            self.state[:7] = copy.deepcopy(joint_pose[:7])
        elif arm=='left':    
            # Assume joint_pose is 8 DoF - 7 for the arm, and 1 for the gripper.
            self.state[7:] = copy.deepcopy(joint_pose[:7])
        elif arm=='both':
            # The Plans were generated as: Left arm, Right arm, left gripper, right gripper.
            # Assume joint_pose is 16 DoF. 7 DoF for left arm, 7 DoF for right arm. (These need to be flipped)., 1 for left gripper. 1 for right gripper.            
            # First right hand. 
            self.state[:7] = joint_pose[7:14]
            # Now left hand. 
            self.state[7:] = joint_pose[:7]
        # Set the joint angles magically. 
        self.environment.set_robot_joint_positions(self.state)

        action = np.zeros((16))
        if gripper:
            # Left gripper is 15. Right gripper is 14. 
            # MIME Gripper values are from 0 to 100 (Close to Open), but we treat the inputs to this function as 0 to 1 (Close to Open), and then rescale to (-1 Open to 1 Close) for Mujoco.
            if arm=='right':
                action[14] = -joint_pose[-1]*2+1
            elif arm=='left':                        
                action[15] = -joint_pose[-1]*2+1
            elif arm=='both':
                action[14] = -joint_pose[15]*2+1
                action[15] = -joint_pose[14]*2+1
            # Move gripper positions.
            self.environment.step(action)

        image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
        return image

    def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif"):

        image_list = []
        for t in range(trajectory.shape[0]):
            new_image = self.set_joint_pose_return_image(trajectory[t])
            image_list.append(new_image)

        if return_gif:
            return image_list
        else:
            imageio.mimsave(os.path.join(gif_path,gif_name), image_list)            

if __name__ == '__main__':
    # end_eff_pose = [0.3, -0.3, 0.09798524029948213, 0.38044099037703677, 0.9228975092885654, -0.021717379118030174, 0.05525572942370394]
    # end_eff_pose = [0.53303758, -0.59997265,  0.09359371,  0.77337391,  0.34998901, 0.46797516, -0.24576358]
    end_eff_pose = np.array([0.64, -0.83, 0.09798524029948213, 0.38044099037703677, 0.9228975092885654, -0.021717379118030174, 0.05525572942370394])
    visualizer = MujocoVisualizer()
    img = visualizer.set_ee_pose_return_image(end_eff_pose, arm='right')
    scipy.misc.imsave('mj_vis.png', img)
    
