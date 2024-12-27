# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP
from models.fields import NPullNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, args, conf_path, dataname, timestamp, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        self.base_exp_dir = os.path.join(self.conf["general.base_exp_dir"], timestamp, dataname)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        self.dataset_np = DatasetNP(self.conf['dataset'], dataname)
        self.dataname = dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')

        self.mode = mode

        # Networks
        self.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)
        self.ChamferDisL1 = ChamferDistanceL1().cuda()
        self.ChamferDisL2 = ChamferDistanceL2().cuda()
        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def update_samples(self,samples, index):
        gradients = self.sdf_network.gradient(samples, index=index).squeeze()  # 5000x3
        sdf_values = self.sdf_network.sdf(samples, index=index)                # 5000x1
        normalized_gradients = F.normalize(gradients, dim=1)                   # 5000x3
        sample_moved = samples - normalized_gradients * sdf_values       
        return sample_moved, gradients, sdf_values        
    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size
        res_step = self.maxiter - self.iter_step
        for iter_i in tqdm(range(res_step)):
            #if(self.iter_step<3000):
            points,samples,_,_, point_gt = self.dataset_np.np_train_data(batch_size)
            # else:
            #     _,_,points,samples,point_gt = self.dataset_np.np_train_data(batch_size)
            samples.requires_grad = True
            sample_history = []
            gradient_history = []
            sdf_value_history = []
            for i in range(3):
                samples_moved,gradients,sdf = self.update_samples(samples, index=i)
                sample_history.append(samples_moved)
                gradient_history.append(gradients)
                sdf_value_history.append(sdf)
            st1_sample_moved, st2_sample_moved, st3_sample_moved = sample_history
            grad_norm1, grad_norm2, grad_norm3 = gradient_history
            st3_sdf_sample = sdf_value_history[-1]
            sdf_surf_loss=torch.mean(st3_sdf_sample**2)
            grad_sim_loss=(1-min(F.cosine_similarity(grad_norm1, grad_norm2, dim=1).mean(),F.cosine_similarity(grad_norm1, grad_norm3, dim=1).mean()))
            focal_l2_loss=self.focal_loss(points,st1_sample_moved,st2_sample_moved)+torch.linalg.norm((points-st3_sample_moved), ord=2, dim=-1).mean()
            loss=focal_l2_loss+1e-3(sdf_surf_loss+grad_sim_loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            #self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.val_freq == 0:
                mc_threshold=[-0.005]
                for  i in range (len(mc_threshold)):
                    self.validate_mesh(resolution=128, threshold=mc_threshold[i], point_gt=point_gt, iter_step=self.iter_step, logger=logger)
            if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                self.save_checkpoint()
            self.iter_step += 1
    def focal_loss(self,st1_points,st1_sample_moved,st2_sample_moved,gamma=2):
        far_dis=torch.linalg.norm((st1_points-st1_sample_moved), ord=2, dim=-1).mean()
        near_dis=torch.linalg.norm((st1_points-st2_sample_moved), ord=2, dim=-1).mean()
        weight_matrix=torch.tensor([far_dis, near_dis], dtype=torch.float32)
        weight_matrix=torch.nn.functional.softmax(weight_matrix,dim=0)
        dynamic_alpha=torch.clamp(weight_matrix[0],max=1/self.maxiter)
        dynamic_beta=torch.clamp((1-weight_matrix[0])**gamma,max=1/self.maxiter)
        focal_loss=dynamic_alpha*far_dis+dynamic_beta*near_dis
        return focal_loss

    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):

        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))
        vertices=mesh.vertices
        vertices*=self.dataset_np.shape_scale
        vertices+=self.dataset_np.shape_center
        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/faust.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--filelist', type=str, default='input.txt')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    data_list = []
    base_dir='/data1/yth/multipull_code/frequency-sdf/'
    with open(os.path.join(base_dir+args.filelist)) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            data_list.append(line)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.mode == 'train':
        for dataname in data_list:
            runner = Runner(args, args.conf, dataname, timestamp)
            runner.train()

