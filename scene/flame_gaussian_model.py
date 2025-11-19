# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


# Innovation 2: Expression-Dependent Appearance Network (EDAN)
class AppearanceNetwork(nn.Module):
    """
    Lightweight MLP that predicts per-vertex appearance modulation based on expression parameters.
    This allows the model to capture expression-dependent lighting and texture changes 
    (e.g., wrinkles, shadows) that cannot be represented by static Gaussian colors.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize to zero so that the network starts with no effect
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None
        self.num_flame_verts = self.flame_model.v_template.shape[0]

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()
        
        # Innovation 2: Initialize Expression-Dependent Appearance Network
        # Input: expression (n_expr) + jaw_pose (3)
        # Output: 3D color offset for each FLAME vertex
        self.appearance_net = None  # Will be initialized after knowing num_verts and data
        self.use_appearance_net = True
        self.appearance_condition_dim = self.n_expr + 3
        self.current_appearance_offset = None

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
            
            # Innovation 2: Initialize appearance network now that we know num_verts
            # Network predicts per-vertex color offsets based on expression
            if self.appearance_net is None:
                input_dim = self.n_expr + 3  # expr + jaw_pose
                # Output: per-vertex RGB offsets (will be propagated to Gaussians via binding)
                output_dim = num_verts * 3
                self.appearance_net = AppearanceNetwork(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=128,
                    num_layers=3
                ).cuda()
                print(f"[Innovation 2] Expression-Dependent Appearance Network initialized: {input_dim} -> {output_dim}")
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)
        
        # Innovation 2: Compute expression-dependent appearance offset
        if self.use_appearance_net and self.appearance_net is not None:
            # Prepare condition input: [expr, jaw_pose]
            curr_expr = flame_param['expr'][[timestep]]  # [1, n_expr]
            curr_jaw = flame_param['jaw_pose'][[timestep]]  # [1, 3]
            condition = torch.cat([curr_expr, curr_jaw], dim=1)  # [1, n_expr + 3]
            
            # Predict per-vertex color offsets [1, num_verts * 3]
            vertex_offsets = self.appearance_net(condition)  # [1, V * 3]
            vertex_offsets = vertex_offsets.reshape(1, self.num_flame_verts, 3)  # [1, V, 3]
            
            # Store for later use in rendering (propagate to Gaussians via binding)
            self.current_appearance_offset = vertex_offsets.squeeze(0)  # [V, 3]
            self.current_gaussian_appearance_delta = self._vertex_offsets_to_gaussians(self.current_appearance_offset)
        else:
            self.current_appearance_offset = None
            self.current_gaussian_appearance_delta = None
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def _vertex_offsets_to_gaussians(self, vertex_offsets):
        """
        Propagate per-vertex appearance offsets to Gaussians via binding.
        Each Gaussian is bound to a face, so we average the offsets of the face's vertices.
        
        Args:
            vertex_offsets: [V, 3] tensor of per-vertex color offsets
            
        Returns:
            [N, 1, 3] tensor of per-Gaussian color offsets
        """
        if vertex_offsets is None or self.binding is None:
            return None
        
        faces = self.flame_model.faces.to(vertex_offsets.device).long()  # [F, 3]
        binding = self.binding.to(vertex_offsets.device).long()  # [N]
        bound_faces = faces[binding]  # [N, 3]
        
        v0_offsets = vertex_offsets[bound_faces[:, 0]]  # [N, 3]
        v1_offsets = vertex_offsets[bound_faces[:, 1]]  # [N, 3]
        v2_offsets = vertex_offsets[bound_faces[:, 2]]  # [N, 3]
        
        gaussian_offsets = (v0_offsets + v1_offsets + v2_offsets) / 3.0  # [N, 3]
        
        return gaussian_offsets.unsqueeze(1)  # [N, 1, 3]
    
    @property
    def get_features(self):
        """
        Override base class to add expression-dependent appearance modulation.
        Returns concatenated [features_dc, features_rest] with dynamic appearance applied to DC component.
        """
        features_dc = self._features_dc
        features_rest = self._features_rest
        
        # Innovation 2: Apply expression-dependent appearance offset
        if self.use_appearance_net and self.current_gaussian_appearance_delta is not None:
            # Add the predicted offset to the DC component (base color)
            # Scale down the offset to prevent overly strong changes
            features_dc = features_dc + self.current_gaussian_appearance_delta * 0.1
        
        return torch.cat((features_dc, features_rest), dim=1)
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def compute_appearance_regularization_loss(self):
        """
        Innovation 2: Regularization loss for appearance network.
        Encourages the predicted appearance offsets to be small and smooth.
        """
        if not self.use_appearance_net or self.current_appearance_offset is None:
            return torch.tensor(0.0, device='cuda')
        
        # L2 regularization on appearance offsets
        return torch.norm(self.current_appearance_offset, p=2) / self.current_appearance_offset.numel()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)
        
        # Innovation 2: Add appearance network to optimizer
        if self.use_appearance_net and self.appearance_net is not None:
            appearance_lr = getattr(training_args, 'appearance_net_lr', 5e-5)
            param_appearance = {'params': self.appearance_net.parameters(), 'lr': appearance_lr, "name": "appearance_net"}
            self.optimizer.add_param_group(param_appearance)
            print(f"[Innovation 2] Appearance network added to optimizer with lr={appearance_lr}")

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
