from visualization.visualize_api import visualize_pair_corr, visualize_reconstructions
from data.point_cloud_db.point_cloud_dataset import PointCloudDataset

from models.sub_models.dgcnn.dgcnn_modular import DGCNN_MODULAR
from models.sub_models.dgcnn.dgcnn import get_graph_feature

import numpy as np

from torch.optim.lr_scheduler import MultiStepLR

from torch_cluster import knn
import pointnet2_ops._ext as _ext
from models.correspondence_utils import get_s_t_neighbors
from models.shape_corr_trainer import ShapeCorrTemplate
from models.metrics.metrics import AccuracyAssumeEye, AccuracyAssumeEyeSoft, uniqueness

from utils import switch_functions

import torch
import torch.nn.functional as F

from models.sub_models.dgcnn.dgcnn import DGCNN as non_geo_DGCNN

from utils.argparse_init import str2bool

# from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds


class GroupingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None

grouping_operation = GroupingOperation.apply
# class GroupingOperation(torch.nn.Module):
#     def __init__(self, nn):
#         super().__init__()
#         self.conv = PointNetConv(nn, add_self_loops=False)

#     def forward(self, x, pos, batch):
#         x_dst = None if x is None else x[idx]
#         x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
#         return x

# grouping_operation = GroupingOperation
class DeepPointCorr(ShapeCorrTemplate):
    
    def __init__(self, hparams, **kwargs):
        """Stub."""
        super(DeepPointCorr, self).__init__(hparams, **kwargs)
        self.encoder = DGCNN_MODULAR(self.hparams, use_inv_features=self.hparams.use_inv_features)

        # self.accuracy_assume_eye = AccuracyAssumeEye()
        # self.accuracy_assume_eye_soft_0p01 = AccuracyAssumeEyeSoft(top_k=int(0.01 * self.hparams.num_points))
        # self.accuracy_assume_eye_soft_0p05 = AccuracyAssumeEyeSoft(top_k=int(0.05 * self.hparams.num_points))


    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()
            
        source_pcd = Pointclouds(points=pos1)
        target_pcd = Pointclouds(points=pos2)

        loss, _ = chamfer_distance(source_pcd, target_pcd)

        # dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        # loss = torch.mean(dist1) + torch.mean(dist2)

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[6, 9], gamma=0.1)
        return [self.optimizer], [self.scheduler]

    def compute_deep_features(self, shape):
        shape["dense_output_features"] = self.encoder.forward_per_point(
            shape['verts'], start_neighs=shape["neigh_idxs"]
        )
        return shape

    def forward_source_target(self, source, target):
        for shape in [source, target]:
            shape = self.compute_deep_features(shape)

        # measure cross similarity
        P_non_normalized = switch_functions.measure_similarity(self.hparams.similarity_init, source["dense_output_features"], target["dense_output_features"])
        
        temperature = None
        P_normalized = P_non_normalized

        # cross nearest neighbors and weights
        source["cross_nn_weight"], source["cross_nn_sim"], source["cross_nn_idx"], target["cross_nn_weight"], target["cross_nn_sim"], target["cross_nn_idx"] =\
            get_s_t_neighbors(self.hparams.k_for_cross_recon, P_normalized, sim_normalization=self.hparams.sim_normalization)

        # cross reconstruction
        source["cross_recon"], source["cross_recon_hard"] = self.reconstruction(source['verts'], target["cross_nn_idx"], target["cross_nn_weight"], self.hparams.k_for_cross_recon)
        target["cross_recon"], target["cross_recon_hard"] = self.reconstruction(target['verts'], source["cross_nn_idx"], source["cross_nn_weight"], self.hparams.k_for_cross_recon)

        return source, target, P_normalized, temperature

    @staticmethod
    def reconstruction(pos, nn_idx, nn_weight, k):
        nn_pos = get_graph_feature(pos.transpose(1, 2), k=k, idx=nn_idx, only_intrinsic='neighs', permute_feature=False)
        nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
        recon = torch.sum(nn_weighted, dim=2)

        recon_hard = nn_pos[:, :, 0, :]
 
        return recon, recon_hard

    def forward_shape(self, shape):
        P_self = switch_functions.measure_similarity(self.hparams.similarity_init, shape["dense_output_features"], shape["dense_output_features"])

        # measure self similarity
        nn_idx = shape['neigh_idxs'][:,:,:self.hparams.k_for_self_recon + 1] if self.hparams.use_euclidiean_in_self_recon else None
        shape["self_nn_weight"], _, shape["self_nn_idx"], _, _, _ = \
            get_s_t_neighbors(self.hparams.k_for_self_recon + 1, P_self, sim_normalization=self.hparams.sim_normalization, s_only=True, ignore_first=True,nn_idx=nn_idx)

        # self reconstruction
        shape["self_recon"], _ = self.reconstruction(shape['verts'], shape["self_nn_idx"], shape["self_nn_weight"], self.hparams.k_for_self_recon)

        return shape, P_self

    @staticmethod
    def batch_frobenius_norm_squared(matrix1, matrix2):
        loss_F = torch.sum(torch.pow(matrix1 - matrix2, 2), dim=[1, 2])

        return loss_F

    def get_perm_loss(self, P, do_soft_max=True):
        batch_size = P.shape[0]

        I = torch.eye(n=P.shape[1], device=P.device)
        I = I .unsqueeze(0).repeat(batch_size, 1, 1)

        if do_soft_max:
            P_normed = F.softmax(P, dim=2)
        else:
            P_normed = P

        perm_loss = torch.mean(self.batch_frobenius_norm_squared(torch.bmm(P_normed, P_normed.transpose(2, 1).contiguous()), I.float()))

        return perm_loss

    @staticmethod
    def get_neighbor_loss(source, source_neigh_idxs, target_cross_recon, k):
        # source.shape[1] is the number of points

        if k < source_neigh_idxs.shape[2]:
            neigh_index_for_loss = source_neigh_idxs[:, :, :k]
        else:
            neigh_index_for_loss = source_neigh_idxs

        source_grouped = grouping_operation(source.transpose(1, 2).contiguous(), neigh_index_for_loss.int()).permute(0, 2, 3, 1)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff ** 2, dim=-1)

        target_cr_grouped = grouping_operation(target_cross_recon.transpose(1, 2).contiguous(), neigh_index_for_loss.int()).permute(0, 2, 3, 1)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target_cross_recon, 2)  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.mean(neighbor_loss_per_neigh)

        return neighbor_loss

    def forward(self, data):

        for shape in ['first', 'second']:
            data[shape]["edge_index"] = [
                knn(data[shape]['verts'][i], data[shape]['verts'][i], self.hparams.num_neighs,)
                for i in range(data[shape]['verts'].shape[0])
            ]
            data[shape]["neigh_idxs"] = torch.stack(
                [data[shape]["edge_index"][i][1].reshape(data[shape]['verts'].shape[1], -1) for i in range(data[shape]['verts'].shape[0])]
            )

        # dense features, similarity, and cross reconstruction
        data['first'], data['second'], data["P_normalized"], data["temperature"] = self.forward_source_target(data['first'], data['second'])

        # cross reconstruction losses
        self.losses[f"source_cross_recon_loss"] = self.hparams.cross_recon_lambda * self.chamfer_loss(data['first']['verts'], data['first']["cross_recon"])
        self.losses[f"target_cross_recon_loss"] =self.hparams.cross_recon_lambda * self.chamfer_loss(data['second']['verts'], data['second']["cross_recon"])

        # self reconstruction
        if self.hparams.use_self_recon:
            _, P_self_source = self.forward_shape(data['first'])
            _, P_self_target = self.forward_shape(data['second'])

            # self reconstruction losses
            data['first']["self_recon_loss_unscaled"] = self.chamfer_loss(data['first']['verts'], data['first']["self_recon"])
            data['second']["self_recon_loss_unscaled"] = self.chamfer_loss(data['second']['verts'], data['second']["self_recon"])

            self.losses[f"source_self_recon_loss"] = self.hparams.self_recon_lambda * data['first']["self_recon_loss_unscaled"]
            self.losses[f"target_self_recon_loss"] = self.hparams.self_recon_lambda * data['second']["self_recon_loss_unscaled"]

        if self.hparams.compute_perm_loss:
            data[f"perm_loss_fwd_unscaled"] = self.get_perm_loss(data["P_normalized"])
            data[f"perm_loss_bac_unscaled"] = self.get_perm_loss(data["P_normalized"].transpose(2, 1).contiguous())

            if self.hparams.perm_loss_lambda > 0.0:
                self.losses[f"perm_loss_fwd"] = self.hparams.perm_loss_lambda * data[f"perm_loss_fwd_unscaled"]
                self.losses[f"perm_loss_bac"] = self.hparams.perm_loss_lambda * data[f"perm_loss_bac_unscaled"]



        if self.hparams.compute_neigh_loss and self.hparams.neigh_loss_lambda > 0.0:
            data[f"neigh_loss_fwd_unscaled"] = \
                self.get_neighbor_loss(data['first']['verts'], data['first']["neigh_idxs"], data['second']["cross_recon"], self.hparams.k_for_cross_recon)
            data[f"neigh_loss_bac_unscaled"] = \
                self.get_neighbor_loss(data['second']['verts'], data['second']["neigh_idxs"], data['first']["cross_recon"], self.hparams.k_for_cross_recon)

            self.losses[f"neigh_loss_fwd"] = self.hparams.neigh_loss_lambda * data[f"neigh_loss_fwd_unscaled"]
            self.losses[f"neigh_loss_bac"] = self.hparams.neigh_loss_lambda * data[f"neigh_loss_bac_unscaled"]

        self.track_metrics(data)

        return data

    def test_step(self, test_batch, batch_idx):
        self.batch=test_batch
        self.hparams.mode = 'test'
        self.hparams.batch_idx=batch_idx
        
        label, pinput1, input2, ratio_list, soft_labels, dist = self.extract_labels_for_test(test_batch)

        source = {'verts': pinput1, "id": self.batch['first']['name']}
        target = {'verts': input2, "id": self.batch['second']['name']}
        batch = {'first': source, 'second': target}
        batch = self(batch)
        p = batch["P_normalized"].clone()


        _ = self.calculate_geodesic_error(label, p, dist, track_dict=self.tracks, hparams=self.hparams)
        _ = self.compute_acc_dpc(label, ratio_list, soft_labels, p, track_dict=self.tracks)
        _ = self.plot_pck(self.tracks)
        # _ = self.compute_acc(label, ratio_list, soft_labels, p,input2,track_dict=self.tracks,hparams=self.hparams)

        self.log_test_step()
        if self.vis_iter():
            self.visualize(batch, mode='test')

        return True


    def visualize(self, batch, mode="train"):
        visualize_pair_corr(self,batch, mode=mode)
        visualize_reconstructions(self,batch, mode=mode)
        
    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = ShapeCorrTemplate.add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False)
        parser = non_geo_DGCNN.add_model_specific_args(parser, task_name, dataset_name, is_lowest_leaf=True)
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False)
        parser.add_argument("--k_for_cross_recon", default=10, type=int, help="number of neighbors for cross reconstruction")

        parser.add_argument("--use_self_recon", nargs="?", default=True, type=str2bool, const=True, help="whether to use self reconstruction")
        parser.add_argument("--k_for_self_recon", default=10, type=int, help="number of neighbors for self reconstruction")
        parser.add_argument("--self_recon_lambda", type=float, default=10.0, help="weight for self reconstruction loss")
        parser.add_argument("--cross_recon_lambda", type=float, default=1.0, help="weight for cross reconstruction loss")

        parser.add_argument("--compute_perm_loss", nargs="?", default=False, type=str2bool, const=True, help="whether to compute permutation loss")
        parser.add_argument("--perm_loss_lambda", type=float, default=1.0, help="weight for permutation loss")

        parser.add_argument("--optimize_pos", nargs="?", default=False, type=str2bool, const=True, help="whether to compute neighbor smoothness loss")
        parser.add_argument("--compute_neigh_loss", nargs="?", default=True, type=str2bool, const=True, help="whether to compute neighbor smoothness loss")
        parser.add_argument("--neigh_loss_lambda", type=float, default=1.0, help="weight for neighbor smoothness loss")
        parser.add_argument("--num_angles", type=int, default=100,)

        parser.add_argument("--use_euclidiean_in_self_recon", nargs="?", default=False, type=str2bool, const=True, help="whether to use self reconstruction")
        parser.add_argument("--use_all_neighs_for_cross_reco", nargs="?", default=False, type=str2bool, const=True, help="whether to use self reconstruction")
        parser.add_argument("--use_all_neighs_for_self_reco", nargs="?", default=False, type=str2bool, const=True, help="whether to use self reconstruction")

        parser.set_defaults(
            optimizer="adam",
            lr=0.0003,
            weight_decay=5e-4,
            max_epochs=300,
            accumulate_grad_batches=2,
            latent_dim=768,
            DGCNN_latent_dim=512,
            bb_size=24,
            num_neighs=27,

            val_vis_interval=20,
            test_vis_interval=20,
        )

        return parser




    def track_metrics(self, data):
        self.tracks[f"source_cross_recon_error"] = self.chamfer_loss(data['first']['verts'], data['first']["cross_recon_hard"])
        self.tracks[f"target_cross_recon_error"] = self.chamfer_loss(data['second']['verts'], data['second']["cross_recon_hard"])


        if self.hparams.use_self_recon:
            self.tracks[f"source_self_recon_loss_unscaled"] = data['first']["self_recon_loss_unscaled"]
            self.tracks[f"target_self_recon_loss_unscaled"] = data['second']["self_recon_loss_unscaled"]


        if self.hparams.compute_neigh_loss and self.hparams.neigh_loss_lambda > 0.0:
            self.tracks[f"neigh_loss_fwd_unscaled"] = data[f"neigh_loss_fwd_unscaled"]
            self.tracks[f"neigh_loss_bac_unscaled"] = data[f"neigh_loss_bac_unscaled"]

        # nearest neighbors hit accuracy
        source_pred = data["P_normalized"].argmax(dim=2)
        target_neigh_idxs = data['second']["neigh_idxs"]

        target_pred = data["P_normalized"].argmax(dim=1)
        source_neigh_idxs = data['first']["neigh_idxs"]

        # uniqueness (number of unique predictions)
        self.tracks[f"uniqueness_fwd"] = uniqueness(source_pred)
        self.tracks[f"uniqueness_bac"] = uniqueness(target_pred)
