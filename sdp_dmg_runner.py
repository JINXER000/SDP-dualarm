
import os
import pathlib
import time
import numpy as np
import copy
import json
import numpy as np
import torch
import dill
import hydra
import cv2
import collections

from sdp.common.pytorch_util import dict_apply
from sdp.workspace.base_workspace import BaseWorkspace
from sdp.policy.base_image_policy import BaseImagePolicy
from sdp.gym_util.video_recording_wrapper import VideoRecorder
from sdp.model.common.rotation_transformer import RotationTransformer

import sys as _sys
_DMG_REPO = "/home/user/yzchen_ws/imitation_learning/dexmimicgen"
if _DMG_REPO not in _sys.path:
    _sys.path.insert(0, _DMG_REPO)
from scripts.robomimic_dmg_wrapper import DMG_env_switchable,to_camel_case, ts_tuple

import robomimic.utils.obs_utils as ObsUtils

def collect_obs(obs_shape_meta, obs_history, t, obs):
    """
    Collect observations from the environment and store them in obs_history.
    Handles both RGB and low-dim observations according to shape_meta.
    RGB images are expected to be in channels-first format (C,H,W).
    """
    for k, v in obs_shape_meta.items():
        tgt_shape = v['shape']
        if len(tgt_shape)==3 and v["type"] == "rgb" and obs[k].shape!= tgt_shape:
            ## resize image
            img_hwc = np.transpose(obs[k].astype(np.float32), (1, 2, 0))
            img_resized = cv2.resize(img_hwc, (tgt_shape[1], tgt_shape[2]), interpolation=cv2.INTER_LINEAR)
            cur_obs = np.transpose(img_resized, (2, 0, 1))  # Convert to channels-first format
        else:
            cur_obs = obs[k].astype(np.float32)
        obs_history[k][t] = cur_obs
    return


def get_seq_obs(obs_history, t, n_obs_steps):
    """
    Get a sequence of observations for the policy.
    If we don't have enough history, pad with the first observation.
    """
    obs_dict_np = dict()
    if t < n_obs_steps - 1:
        # Pad with first observation if we don't have enough history
        for k, v in obs_history.items():
            obs_dict_np[k] = np.array([v[0]] * n_obs_steps, dtype=np.float32)
            obs_dict_np[k][n_obs_steps-t-1:n_obs_steps] = v[0:t+1]
    else:
        # Get the last n_obs_steps observations
        for k, v in obs_history.items():
            obs_dict_np[k] = v[t-n_obs_steps+1:t+1]
    return obs_dict_np




class SDP_DMG_Evaluator(DMG_env_switchable):
    def __init__(self, checkpoint_dict, output, max_timesteps,\
                 num_inference_steps, with_planning= False,  fps = 10, crf = 22, record = False, render_obs_key = "robot0_eye_in_hand_image"):
        self.max_timesteps = max_timesteps
        self.checkpoint_dict = checkpoint_dict
        self.output = output
        self.num_inference_steps = num_inference_steps

        self.with_planning = with_planning  
        
        ## config image recording
        if record:
            self.render_obs_key = render_obs_key
            self.video_recorder = VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        )
            save_dir = output
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cur_time = time.strftime("%d_%H.%M.%S", time.localtime())
            self.file_path = os.path.join(save_dir,  'test_' +cur_time +'.mp4')
            print(f"{self.render_obs_key} will be saved to: {self.file_path}")
        else:
            self.file_path = None
            self.video_recorder = None

        self.env_initialized = False

    def set_bc_controller(self):
        self.update_controllers(controller_name = "OSC_POSE", abs_action = self.lfd_abs_action)
       

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def initialize_env(self, skill, env_options):
        self.cur_env_name = skill
            
        self.load_checkpoint(env_options)        
        self.ts = self.reset_all()

    def load_checkpoint(self, env_options):
        # load checkpoint
        payload = torch.load(open(self.checkpoint_dict[self.cur_env_name], 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=self.output)
        workspace: BaseWorkspace
        # in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
        if "model" not in workspace.__dict__.keys():
            workspace.model = hydra.utils.instantiate(cfg.policy)
        if "ema_model" not in workspace.__dict__.keys() and cfg.training.use_ema:
            workspace.ema_model = copy.deepcopy(workspace.model)
        if "optimizer" not in workspace.__dict__.keys():
            workspace.optimizer = hydra.utils.instantiate(
                cfg.optimizer, workspace.model.parameters()
            )
        workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

        # # get policy from workspace
        # if 'diffusion' in cfg.name:
        ## diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        self.device = torch.device('cuda')
        policy.eval().to(self.device)

        ## set inference params
        policy.num_inference_steps = self.num_inference_steps #16 # [DDIM inference iterations]
        # else:
        #     raise RuntimeError("Unsupported policy type: ", cfg.name)
        
        self.policy= policy
        # hyper-parameters
        ## observation
        self.obs_shape_meta = cfg.task.shape_meta.obs

        ## setup robomimic observation
        modality_mapping = collections.defaultdict(list)
        for key, attr in self.obs_shape_meta.items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        ## setup action type
        self.action_dim = cfg.task.shape_meta.action.shape
        if self.action_dim[0] > 14:
            self.lfd_abs_action = True
            self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        else:
            self.lfd_abs_action = False
            self.rotation_transformer = None


        ## multi-step params for policy
        self.query_cycle = cfg.n_action_steps
        self.n_obs_steps = cfg.n_obs_steps

        ## setup environment
        env_name = to_camel_case(self.cur_env_name)
        ## TODO： switch between different skills.
        if self.env_initialized:
            raise NotImplementedError("policy switching is not supported yet")
        
        np.random.seed(int(time.time()))
        super().__init__(env_name, env_options)
        self.env_initialized = True

        

    def reset_ts(self, with_planning = False):
        # if with_planning:
        #     raise NotImplementedError("Planning is not implemented yet")
        # else:
        self.raw_obs = self.env.reset()
        self.obs = self.get_observation(self.raw_obs)
        init_ts = ts_tuple(self.obs, 0, False, {})
        return init_ts
    
    def step_ts(self, action):
        self.raw_obs, reward, done, info = self.env.step(action)
        self.obs = self.get_observation(self.raw_obs)
        info["is_success"] = self.is_success()
        return ts_tuple(self.obs, reward, done, info)

    def reset_all(self):
        ts = self.reset_ts(with_planning=self.with_planning)
            
        ## obs history for extracting multi-step obs
        self.obs_history = dict()
        for key in self.obs_shape_meta.keys():
            self.obs_history[key] = np.zeros(
                (self.max_timesteps, *self.obs_shape_meta[key].shape),
                dtype=np.float32
            )
        self.t = 0
        return ts

    def record_frame(self, obs):

        if self.video_recorder is not None:
            try:
                if not self.video_recorder.is_ready():
                    self.video_recorder.start(self.file_path)

                img = np.moveaxis(obs[self.render_obs_key], 0, -1)
                frame = (img * 255).astype(np.uint8) 
                
                self.video_recorder.write_frame(frame)
            except Exception as e:
                print(f"Warning: Failed to record video frame: {e}")
                import traceback
                traceback.print_exc()

    def replay_tamp_step(self, total_action):
        # self.ts = self.replay_tamp_step(total_action)

        start = time.time()

        self.ts = self.step_ts(total_action)
        self.env.render()
        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / self.max_framerate - elapsed
        if diff > 0:
            time.sleep(diff)
            
        self.record_frame(self.ts.observation)
        return self.ts
    # def get_mj_pc_dict(self, **kwargs):
    #     return self.env.save_mj_observation(**kwargs)

    def inference_once(self, render = True):
        if self.t >= self.max_timesteps:
            return True
        with torch.inference_mode():
            # process previous ts
            obs = self.ts.observation
            collect_obs(self.obs_shape_meta, self.obs_history, self.t, obs)
            obs_dict_np = get_seq_obs(self.obs_history, self.t, self.n_obs_steps)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            # query policy to extract action: (B=1, Da)
            # t0 = time.perf_counter()
            if self.t % self.query_cycle == 0:
                action_dict = self.policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                self.np_action_seq = np_action_dict['action'][0] # T,Da

                if self.lfd_abs_action:
                    self.np_action_seq = self.undo_transform_action(self.np_action_seq)

            action = self.np_action_seq[self.t % self.query_cycle]
            # t1 = time.perf_counter()

            self.ts = self.step_ts(action)

            self.t += 1

        if render:
            self.env.render()

        self.record_frame(obs)

        return self.ts.done

    def exit(self):
     
        if self.video_recorder is not None:
            try:
                self.video_recorder.stop()
                print(f"Video saved to: {self.file_path}")
                
                # Verify the video file was created and is valid
                if os.path.exists(self.file_path):
                    file_size = os.path.getsize(self.file_path)
                    if file_size > 0:
                        print(f"Video file created successfully. Size: {file_size} bytes")
                        
                        # Check if the video needs to be converted to proper MP4 format
                        import subprocess
                        try:
                            # Use ffprobe to check the container format
                            result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', self.file_path], 
                                                  capture_output=True, text=True)
                            if result.returncode == 0:
                                # Video is valid, try to ensure it's properly formatted
                                temp_path = self.file_path.replace('.mp4', '_temp.mp4')
                                subprocess.run(['ffmpeg', '-i', self.file_path, '-c', 'copy', '-f', 'mp4', temp_path], 
                                             capture_output=True)
                                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                    os.replace(temp_path, self.file_path)
                                    print("Video converted to proper MP4 format")
                        except Exception as e:
                            print(f"Warning: Could not post-process video: {e}")
                    else:
                        print("Warning: Video file is empty!")
                else:
                    print("Warning: Video file was not created!")
                    
            except Exception as e:
                print(f"Error stopping video recorder: {e}")
        return self.file_path



def wrapper_test():
    output = './data/eval/transfer_cup/'
    checkpoint_dict = {\
        'two_arm_three_piece_assembly': \
              'data/outputs/two_arm_three_piece_assembly/epoch=0440-test_mean_score=1.000.ckpt',
        # 'two_arm_threading': \
        # 'data/outputs/two_arm_threading/epoch=1350-test_mean_score=0.500.ckpt',
            # 'data/outputs/two_arm_threading/latest.ckpt',
                       }
    env_names = list(checkpoint_dict.keys())

    max_timesteps = 500
    num_inference_steps = 10

    env_runer = SDP_DMG_Evaluator(checkpoint_dict, output, max_timesteps, num_inference_steps, record= True)
    

    for skill in env_names:

        ## hardcode ennv_options
        env_options = {}
        # env_options["env_name"] = skill
        # env_options["env_configuration"] = env_configuration
        env_options["robots"] = ['Panda', 'Panda']
        env_options["camera_names"] =["sideview","agentview", "birdview", "frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"]
        env_options["camera_heights"] = 168
        env_options["camera_widths"] =  168
        # env_options["has_offscreen_renderer"] = True
        # env_options["use_camera_obs"] = True
        # env_options["camera_depths"] = True
        env_options["camera_segmentations"] = "instance"
        env_options['output_all_pcds'] = True
        
        env_runer.initialize_env(skill, env_options)
        for i in range(max_timesteps):
            done = env_runer.inference_once()
            task_success = env_runer.handle_rewards()
            if task_success:
                print('Task completed!')
                break
            # dp.append_image()

    env_runer.exit()


if __name__ == '__main__':
    # main()
    wrapper_test()
