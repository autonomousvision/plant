import os
import torch

from carla_agent_files.data_agent_boxes import DataAgent


def get_entry_point():
    return 'ExplainabilityAgent'

LOAD_CKPT_PATH = os.environ.get('LOAD_CKPT_PATH', None)
SAVE_GIF = os.getenv("SAVE_GIF", 'False').lower() in ('true', '1', 't')


class ExplainabilityAgent(DataAgent):
    def setup(self, path_to_conf_file, route_index=None, cfg=None, exec_or_inter=None):
        
        self.epoch = 0
        self.cfg = cfg
        self.args = {}

        super().setup(path_to_conf_file, route_index, cfg, exec_or_inter)

        print(f'Loading model from {LOAD_CKPT_PATH}')
        print(f'Saving gif: {SAVE_GIF}')
        
        if cfg.exec_model == 'PlanT':
            from carla_agent_files.PlanT_agent import PlanTAgent as Exec_Agent
        elif cfg.exec_model == 'Expert':
            from carla_agent_files.autopilot import AutoPilot as Exec_Agent
        else:
            print(f'exec_model {cfg.exec_model} not implemented. Please choose from (PlanT, Expert)')
            raise NotImplementedError
            
        if cfg.inter_model == 'PlanT':
            from carla_agent_files.PlanT_agent import PlanTAgent as Inter_Agent
        else:
            print(f'inter_model {cfg.inter_model} not implemented. Please choose from (PlanT)')
            raise NotImplementedError
            
    
        self.interAgent = Inter_Agent(cfg.inter_agent_config, route_index, cfg, 'inter')
        self.execAgent = Exec_Agent(cfg.exec_agent_config, route_index, cfg, 'exec')


    def _init(self):
        self.interAgent._global_plan_world_coord = self._global_plan_world_coord
        self.execAgent._global_plan_world_coord = self._global_plan_world_coord
        self.interAgent._global_plan = self._global_plan
        self.execAgent._global_plan = self._global_plan
        self.initialized = True
        
        
    def sensors(self):
        result = super().sensors()

        if SAVE_GIF == True:
            result += [
                    # {	
                    #     'type': 'sensor.camera.rgb',
                    #     'x': 1.3, 'y': 0.0, 'z':40,
                    #     'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    #     'width': 500, 'height': 500, 'fov': 90,
                    #     'id': 'spec'
                    #     },
                        {
                        'type': 'sensor.camera.rgb',
                        'x': -9, 'y': 0.0, 'z':9,
                        'roll': 0.0, 'pitch': -30.0, 'yaw': 0.0,
                        'width': 960, 'height': 540, 'fov': 120,
                        # 'width': 1280, 'height': 720, 'fov': 120,
                        # 'width': 1920, 'height': 1080, 'fov': 120,
                        'id': 'rgb_back'
                        },
                        {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': -9, 'y': 0.0, 'z':9,
                        'roll': 0.0, 'pitch': -30.0, 'yaw': 0.0,
                        'width': 960, 'height': 540, 'fov': 120,
                        # 'width': 1280, 'height': 720, 'fov': 120,
                        # 'width': 1920, 'height': 1080, 'fov': 120,
                        'id': 'sem_back'
                        }
                    ]

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        if not self.initialized:
            self._init()
        
        # run inter agent to get ids of topk vehicles (topk highest attention score)
        keep_vehicle_ids, keep_all_ids = self.interAgent.run_step(input_data, timestamp, sensors=sensors)
        # print(len(keep_vehicle_ids))
        
        # run exec agent with masked vehicles (only show topk vehicles to agent)
        if self.cfg.exec_model == 'Expert':
            self.control = self.execAgent.run_step(input_data, timestamp, keep_ids=keep_vehicle_ids, sensors=sensors)
        else:
            self.control = self.execAgent.run_step(input_data, timestamp, keep_ids=keep_all_ids, sensors=sensors)

        return self.control
        

    def destroy(self):
        
        self.interAgent.destroy()
        self.execAgent.destroy()

        super().destroy()
        self.epoch += 1
        print('destroyed')