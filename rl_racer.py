from argparse import ArgumentParser
from threading import Thread
import time
import math

import cv2
import numpy as np
import torch
import akro
import gym
from scipy.spatial.transform.rotation import Rotation

import garage.torch.utils as tu
from garage.envs import GarageEnv, normalize
from garage import wrap_experiment
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.torch.algos import VPG
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

# airsimneurips API
import airsimneurips as airsim
from baselines.baseline_racer import BaselineRacer

# GPU/CPU support
if torch.cuda.is_available():
    tu.set_gpu_mode(True)
else:
    tu.set_gpu_mode(False)

parser = ArgumentParser()
parser.add_argument('--level_name', type=str,
                    choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard",
                             "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3", "Final_Tier_1",
                             "Final_Tier_2", "Final_Tier_3"], default="Soccer_Field_Easy")
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=True)
parser.add_argument('--race_tier', type=int, choices=[1, 2, 3], default=1)
args = parser.parse_args()

# video recorder
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('./fpv_cam.avi', fourcc, 20.0, (320, 240))


class RLRacer(BaselineRacer):
    '''Racing drone with reinforcement learning used as a planner'''

    def __init__(self, agent, opponent):
        print("Creating RLRacer object...")
        self.agent = agent
        self.opponent = opponent
        super(RLRacer, self).__init__()
        self.img_rgb = None
        self.odom = None

    def image_callback(self, get_data=False):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if self.img_rgb is None:
            self.img_rgb = img_rgb
        if self.viz_image_cv2 and not get_data:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)

    def odometry_callback(self):
        # get uncompressed fpv cam image
        agent_state = self.airsim_client_odom.getMultirotorState(vehicle_name=self.agent)
        opponent_state = self.airsim_client_odom.getMultirotorState(vehicle_name=self.opponent)
        # agent_state = self.airsim_client_odom.simGetGroundTruthKinematics(vehicle_name=self.agent)
        # opponent_state = self.airsim_client_odom.simGetGroundTruthKinematics(vehicle_name=self.opponent)
        # in world
        agent_position = agent_state.kinematics_estimated.position
        agent_orientation = agent_state.kinematics_estimated.orientation
        agent_linear_velocity = agent_state.kinematics_estimated.linear_velocity
        agent_angular_velocity = agent_state.kinematics_estimated.angular_velocity
        agent_timestamp = agent_state.timestamp

        opponent_position = opponent_state.kinematics_estimated.position
        opponent_orientation = opponent_state.kinematics_estimated.orientation
        opponent_linear_velocity = opponent_state.kinematics_estimated.linear_velocity
        opponent_angular_velocity = opponent_state.kinematics_estimated.angular_velocity
        opponent_timestamp = opponent_state.timestamp
        if self.odom is None:
            self.odom = {'agent': [agent_timestamp, agent_position, agent_orientation,
                                   agent_linear_velocity, agent_angular_velocity],
                         'opponent': [opponent_timestamp, opponent_position, opponent_orientation,
                                      opponent_linear_velocity, opponent_angular_velocity, opponent_timestamp]}

    def stop_image_callback_thread(self):
        ret = None
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            ret = self.image_callback_thread.join()
            print("Stopped image callback thread.")
        return ret

    def stop_odometry_callback_thread(self):
        ret = None
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            ret = self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")
        return ret

    def pause_sim(self):
        if not self.airsim_client.simIsPaused():
            self.airsim_client.simPause()

    def unpause_sim(self):
        if self.airsim_client.simIsPaused():
            self.airsim_client.simUnPause()

    def get_img_rgb(self):
        return self.img_rgb

    def get_odom(self):
        return self.odom

    def train(self, ctxt=None, seed=42):
        env = GarageEnv(normalize(RacingEnv(self)))
        setattr(env, 'name', self.level_name)
        set_seed(seed)
        runner = LocalRunner()
        policy = GaussianMLPPolicy(env_spec=env.spec,
                                   hidden_sizes=[256, 256, 128],
                                   hidden_nonlinearity=torch.tanh,
                                   output_nonlinearity=None)
        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                  hidden_sizes=[128, 128, 64],
                                                  hidden_nonlinearity=torch.tanh,
                                                  output_nonlinearity=None
                                                  )

        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   value_function=value_function,
                   max_path_length=500,
                   discount=0.99,
                   center_adv=False)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=1)

    def test(self):
        pass

    def reset(self):
        self.airsim_client.simPause()
        self.airsim_client.reset()
        # start again
        drone_names = ['drone_1', 'drone_2']
        print("Starting up...")
        for drone_name in drone_names:
            if drone_name == 'drone_1':
                self.airsim_client.enableApiControl(vehicle_name=drone_name)
                self.airsim_client.arm(vehicle_name=drone_name)
                traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track=5.0, kd_cross_track=0.0,
                                                                   kp_vel_cross_track=3.0, kd_vel_cross_track=0.0,
                                                                   kp_along_track=0.4, kd_along_track=0.0,
                                                                   kp_vel_along_track=0.04, kd_vel_along_track=0.0,
                                                                   kp_z_track=2.0, kd_z_track=0.0,
                                                                   kp_vel_z=0.4, kd_vel_z=0.0,
                                                                   kp_yaw=3.0, kd_yaw=0.1)

            self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=drone_name)

        time.sleep(0.2)
        self.airsim_client.simUnPause()
        # self.racer.start_race(tier=self.tier)
        self.airsim_client.simResetRace()
        self.airsim_client.simStartRace()


class RacingEnv(gym.Env):
    '''OpenAI Gym environment for drone racing.
    This environment should be created when running train or test for a MetaRLRacer'''

    def __init__(self, racer, **kwargs):
        print("Creating OpenAI-gym environment...")
        # assert isinstance(racer, RLRacer)
        self.racer = racer
        print("Creating client for environment...")
        self.env_client = airsim.MultirotorClient()
        self.env_client.confirmConnection()
        if self.racer.gate_poses_ground_truth is None:
            self.racer.get_ground_truth_gate_poses()
        # create gym environment for RL experiments
        super(RacingEnv, self).__init__()
        param_defaults = {
            # MDP dynamics
            'G': None,
            'vxmax': 10,
            'vymax': 10,
            'vzmax': 10,
            'T': 100,
            'k': 1.5,  # distance weight
            'C': 10,  # obstacle crash cost
            'alpha': 1e-2,  # velocity weight (penalizes small velocities)
            'beta': 0.5,  # weights penalty for lagging behind in the race
            'gamma': 2,  # weight gate reward
            'loss_reward': -1000,
            'win_reward': 1000,
            'img_height': 240,
            'img_width': 320,
            'img_channels': 3,
            'tier': 1,
            'dt': 0.3  # time between controls and state observations
        }
        # Keep track of state of the environment
        self._state = Observation(None, None, gates_passed={'agent': 0, 'opponent': 0},
                                  gate_pose=self.racer.gate_poses_ground_truth[0])
        # set all necessary attributes to desired default values
        for key, value in param_defaults.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def observation_space(self):
        min_bounds = np.array([-np.inf, -np.inf, -np.inf,  # x_self_low, y_self_low, z_self_low
                               -np.inf, -np.inf, -np.inf, -np.inf,  # quaternion_low
                               ### drone 1 velocities
                               # vx_self_low, vy_self_low, vz_self_low (linear)
                               -np.inf, -np.inf, -np.inf,
                               # wx_self_low, wy_self_low, wz_self_low (angular)
                               -np.inf, -np.inf, -np.inf,
                               # gates passed
                               0,
                               ### opponent pose
                               -np.inf, -np.inf, -np.inf,  # x_opp_low, y_opp_low, z_opp_low
                               -np.inf, -np.inf, -np.inf, -np.inf,  # quaternion_low
                               ### drone 2 velocities
                               # vx_opp_low, vy_opp_low, vz_opp_low (linear)
                               -np.inf, -np.inf, -np.inf,
                               # wx_opp_low, wy_opp_low, wz_opp_low (angular)
                               -np.inf, -np.inf, -np.inf,
                               # gates passed
                               0,
                               # gate pose
                               -np.inf, -np.inf, -np.inf,
                               -np.inf, -np.inf, -np.inf, -np.inf
                               ])

        max_bounds = np.array([np.inf, np.inf, np.inf,  # x_self_high, y_self_high, z_self_high
                               np.inf, np.inf, np.inf, np.inf,  # quaternion_high
                               ### drone 1 velocities
                               # vx_self_high, vy_self_high, vz_self_high (linear)
                               np.inf, np.inf, np.inf,
                               # wx_self_high, wy_self_high, wz_self_high (angular)
                               np.inf, np.inf, np.inf,
                               # gates passed
                               self.G,
                               ### opponent pose
                               np.inf, np.inf, np.inf,  # x_opp_high, y_opp_high, z_opp_high
                               np.inf, np.inf, np.inf, np.inf,  # quaternion_high
                               ### drone 2 velocities
                               # vx_opp_high, vy_opp_high, vz_opp_high (linear)
                               np.inf, np.inf, np.inf,
                               # wx_opp_high, wy_opp_high, wz_opp_high (angular)
                               np.inf, np.inf, np.inf,
                               # gates passed
                               self.G,
                               # gate pose
                               np.inf, np.inf, np.inf,
                               np.inf, np.inf, np.inf, np.inf
                               ])

        img_space = akro.Box(low=0, high=255, shape=(self.img_height, self.img_width, self.img_channels),
                             dtype=np.uint8)

        return akro.Box(low=min_bounds, high=max_bounds, dtype=np.float32)
        # return img_space.concat(akro.Box(low=min_bounds, high=max_bounds, dtype=np.float32))

    @property
    def action_space(self):
        min_bounds = np.array([-self.vxmax, -self.vymax, -self.vzmax,  # vx, vy, vz
                               -np.pi, -np.pi, -np.pi,  # roll, pitch, yaw
                               0])  # min_throttle
        max_bounds = np.array([self.vxmax, self.vymax, self.vzmax,
                               np.pi, np.pi, np.pi,
                               1])
        return akro.Box(low=min_bounds, high=max_bounds, dtype=np.float32)

    def observe(self):
        # get image and odometry data
        self.racer.img_rgb = None
        self.racer.odom = None
        while self.racer.img_rgb is None or self.racer.odom is None:
            # let values become populated before proceeding
            time.sleep(0.02)
        img_rgb = self.racer.get_img_rgb()
        odom = self.racer.get_odom()

        # get score and gate data
        gates_passed = self._state.gates_passed
        for drone in ['agent', 'opponent']:
            curr_position_linear = odom[drone][1]
            next_gate_idx = self._state.gates_passed[drone]
            dist_from_next_gate = math.sqrt(
                (curr_position_linear.x_val - self.racer.gate_poses_ground_truth[next_gate_idx].position.x_val) ** 2
                + (curr_position_linear.y_val - self.racer.gate_poses_ground_truth[
                    next_gate_idx].position.y_val) ** 2
                + (curr_position_linear.z_val - self.racer.gate_poses_ground_truth[
                    next_gate_idx].position.z_val) ** 2)

            if dist_from_next_gate < 1e-2:
                last_gate_passed_idx = next_gate_idx
                next_gate_idx += 1
                gates_passed[drone] = next_gate_idx

            if drone == 'agent':
                next_gate_pose = self.racer.gate_poses_ground_truth[next_gate_idx]

        video_writer.open('./fpv_cam.avi', fourcc, 20.0, (320, 240))
        video_writer.write(img_rgb)
        obs = Observation(img_rgb=img_rgb, odom=odom, gates_passed=gates_passed.copy(),
                          gate_pose=next_gate_pose)
        self._state = obs

        print(obs)
        return obs

    def reset(self, imshow=True):
        print("Resetting environment...")
        # reset
        self.racer.reset()
        print("Taking off...")
        # self.racer.initialize_drone()
        self.env_client.takeoffAsync(vehicle_name=self.racer.agent)
        print("Done. Starting image+odom callbacks...")
        self.racer.get_ground_truth_gate_poses()
        # time.sleep(1)

        if not self.racer.airsim_client.simIsPaused():
            self.racer.airsim_client.simPause()
            print("Simulator paused.")

        obs = self.observe()

        return obs.to_numpy_array()

    def render(self):
        pass

    def step(self, action, imshow=True):
        # action = (vx, vy, vz, roll, pitch, yaw_rate, throttle)
        # resume simulation if paused and send control input

        self.racer.airsim_client.simPause()
        self.racer.airsim_client.cancelLastTask(vehicle_name=self.racer.agent)
        self.racer.airsim_client.moveByRollPitchYawThrottleAsync(roll=action[3].item(), pitch=action[4].item(),
                                                                 yaw=action[5].item(), throttle=action[6].item(),
                                                                 duration=10)
        self.racer.airsim_client.moveByVelocityAsync(vx=action[0].item(), vy=action[1].item(), vz=action[2].item(),
                                                     duration=10)
        t = self.dt / 10.
        self.racer.airsim_client.simContinueForTime(duration=t)

        next_observation = self.observe()
        print(next_observation)
        reward, done = self.reward_function(action, next_observation)
        return next_observation.to_numpy_array(), reward, done, None

    def reward_function(self, action, observed_state):
        """Reward is the same as negative level cost"""
        collision_info = self.racer.airsim_client.simGetCollisionInfo(vehicle_name=self.racer.agent)
        next_gate_pose = self.racer.gate_poses_ground_truth[0]
        agent_position = observed_state.odom['agent'][1].to_numpy_array()
        opponent_position = observed_state.odom['opponent'][1].to_numpy_array()
        gate_position = self._state.gate_pose.position.to_numpy_array()

        E_xu = self.k * np.linalg.norm(agent_position - gate_position) + \
               self.alpha / (np.linalg.norm(observed_state.odom['agent'][3].to_numpy_array()) + 1e-3) + \
               np.sinh(self.beta * (observed_state.gates_passed['opponent'] - observed_state.gates_passed['agent']))

        G_xu = self.gamma * 2 * observed_state.gates_passed['agent'] + \
               self.alpha / (np.linalg.norm(observed_state.odom['agent'][3].to_numpy_array()) + 1e-3) + \
               np.sinh(self.beta * (observed_state.gates_passed['opponent'] - observed_state.gates_passed['agent']))

        if not collision_info.has_collided:
            # if gate not passed in observed_state and race is not over:
            if observed_state.gates_passed['agent'] != self.G and observed_state.gates_passed['opponent'] != self.G:
                if observed_state.gates_passed['agent'] == self._state.gates_passed['agent']:  # gate not passed thru
                    reward = -E_xu
                else:  # Agent passes through gate
                    reward = -G_xu
                done = False
            else:  # A drone has finished the race
                if observed_state.gates_passed['agent'] == self.G:  # agent wins
                    reward = self.win_reward
                elif observed_state.gates_passed['opponent'] == self.G:  # opponent wins
                    reward = self.loss_reward
                done = True
        else:  # collision occurs
            # if collision with drone
            # if self.env_client.simIsRacerDisqualified(vehicle_name=self.racer.agent):
            # drone is disqualified
            #     reward = self.loss_reward
            #     done = True
            if collision_info.object_name == 'drone_2':
                # drone wins by disqualification of opponent
                reward = self.loss_reward
                done = True
            else:  # if collision with object
                reward = (-self.C) + (-E_xu)
                done = False

        return reward, done


class Observation:
    def __init__(self, img_rgb, odom, gates_passed, gate_pose):
        """
        odom = {'agent':    [agent_timestamp, agent_position, agent_orientation,
                             agent_linear_velocity, agent_angular_velocity],
                'opponent': [opponent_timestamp, opponent_position, opponent_orientation,
                            opponent_linear_velocity, opponent_angular_velocity, opponent_timestamp]}
        """
        self.img_rgb = img_rgb
        self.odom = odom
        self.gates_passed = gates_passed
        self.gate_pose = gate_pose

    def __str__(self):
        r = Rotation.from_quat(self.odom['agent'][2].to_numpy_array())
        orientation = r.as_euler('zyx', degrees=True)

        outstr = ''
        outstr += "Timestamp: {}\nAgent coords: {}\nAgent orientation (yaw-pitch-roll): {}\n" \
            .format(self.odom['agent'][0], self.odom['agent'][1], orientation)

        r = Rotation.from_quat(self.odom['opponent'][2].to_numpy_array())
        orientation = r.as_euler('zyx', degrees=True)

        outstr += "Timestamp: {}\nOpponent coords: {}\nOpponent orientation (yaw-pitch-roll): {}\n" \
            .format(self.odom['opponent'][0], self.odom['opponent'][1], orientation)
        return outstr

    def imshow(self):
        cv2.imshow('observed_img', self.img_rgb)

    def to_numpy_array(self):
        return np.block([[self.odom['agent'][1].to_numpy_array().reshape(1, -1),
                          self.odom['agent'][2].to_numpy_array().reshape(1, -1),
                          self.odom['agent'][3].to_numpy_array().reshape(1, -1)],  # row 1
                         [self.odom['opponent'][1].to_numpy_array().reshape(1, -1),
                          self.odom['opponent'][2].to_numpy_array().reshape(1, -1),
                          self.odom['opponent'][3].to_numpy_array().reshape(1, -1)]  # row 2
                         ])


def main():
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    agent, opponent = 'drone_1', 'drone_2'
    rl_racer = RLRacer(agent='drone_1', opponent='drone_2')
    rl_racer.load_level(args.level_name)
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3
    rl_racer.start_race(args.race_tier)
    rl_racer.initialize_drone()
    rl_racer.takeoff_with_moveOnSpline()
    rl_racer.get_ground_truth_gate_poses()  # after this is called, create environment
    rl_racer.start_image_callback_thread()
    rl_racer.start_odometry_callback_thread()

    # rl_racer.pause_sim()
    # race_env = normalize(RacingEnv(rl_racer))
    # race_env = GarageEnv(env=race_env)
    race_env = RacingEnv(rl_racer)
    T = 0
    while True:
        if T == 40:
            race_env.reset()
            video_writer.release()
            cv2.destroyAllWindows()
            break
        race_env.step(action=race_env.action_space.sample())
        T += 1

    if args.train:
        pass
        # rl_racer.train(race_env)
        # debug_my_algorithm(ctxt=None, env=race_env)

    if args.test:
        # test the trained policy and report results
        rl_racer.test()
    rl_racer.stop_odometry_callback_thread()
    rl_racer.stop_image_callback_thread()

# @wrap_experiment
# def debug_my_algorithm(ctxt=None):
#     set_seed(99)
#     runner = LocalRunner(ctxt)
#     policy = GaussianMLPPolicy(env.spec)
#     algo = VPG(env.spec, policy)
#     algo.to()
#     runner.setup(algo, env)
#     runner.train(n_epochs=499, batch_size=4000)


if __name__ == "__main__":
    main()
