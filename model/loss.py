import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


def z_score_normalize(features: torch.Tensor):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / std

def multiply_off_diagonal(c_diff, lambda_value):
    """
    Multiply the off-diagonal elements of a dxd matrix by lambda.
    
    Parameters:
    c_diff (torch.Tensor): The input square matrix (d x d).
    lambda_value (float): The scalar to multiply the off-diagonal elements by.
    
    Returns:
    torch.Tensor: The modified matrix with off-diagonal elements multiplied by lambda.
    """
    # Create a mask for the off-diagonal elements
    d = c_diff.size(0)
    mask = ~torch.eye(d, dtype=bool, device=c_diff.device)  # Off-diagonal mask
    
    # Multiply off-diagonal elements by lambda
    c_diff[mask] *= lambda_value
    
    return c_diff

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:

            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale=np.log(1 / 0.07), output_dict=False, *args, **kwargs):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            diagonal_weight:float = 0.0,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir
        self.diagonal_weight = diagonal_weight

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels
    
    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits
    
    def _loss(self, image_features, text_features, extra_text_features, logit_scale, logit_bias=None, logits_per_text=None, negative_only=False):
        # breakpoint()

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
      
        if logits_per_text is not None:
            logits = logits_per_text
        else:
            logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)

        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0]
        )

        if labels.shape!= logits.shape:
            breakpoint()

        loss = F.logsigmoid(labels * logits) # N, N


        with torch.no_grad():
            diag_loss = torch.diagonal(loss).sum()
            positive_loss = -diag_loss/(image_features.shape[0]*image_features.shape[0])
            negative_loss = -(loss.sum() - diag_loss) / (image_features.shape[0]*image_features.shape[0])

        output = {"positive_loss": positive_loss, "negative_loss": negative_loss}

      
        #Solution 2: images contrast to all extra text features
        if extra_text_features is not None:
            extra_text_features = F.normalize(extra_text_features, p=2, dim=-1)
            extra_logits = self.get_logits(image_features, extra_text_features, logit_scale, logit_bias)
            extra_loss = F.logsigmoid(labels * extra_logits)
            output["extra_loss"] = -torch.mean(extra_loss)
            loss = torch.cat([loss, extra_loss], dim=1)
        

        output["contrastive_loss"] = -torch.mean(loss)

        return output



    def forward(self, image_features, text_features, extra_text_features, logit_scale, logit_bias, logits_per_text=None, output_dict=False, **kwargs):
       
        loss = self._loss(image_features, text_features, extra_text_features, logit_scale, logit_bias, logits_per_text)
        # if self.world_size > 1:
        #     # exchange text features w/ neighbour world_size - 1 times
        #     right_rank = (self.rank + 1) % self.world_size
        #     left_rank = (self.rank - 1 + self.world_size) % self.world_size
        #     if self.bidir:
        #         text_features_to_right = text_features_to_left = text_features
        #         num_bidir, remainder = divmod(self.world_size - 1, 2)
        #         for i in range(num_bidir):
        #             text_features_recv = neighbour_exchange_bidir_with_grad(
        #                 left_rank,
        #                 right_rank,
        #                 text_features_to_left,
        #                 text_features_to_right,
        #             )

        #             for f in text_features_recv:
        #                 loss += self._loss(
        #                     image_features,
        #                     f,
        #                     logit_scale,
        #                     logit_bias,
        #                     negative_only=True,
        #                 )
        #             text_features_to_left, text_features_to_right = text_features_recv

        #         if remainder:
        #             text_features_recv = neighbour_exchange_with_grad(
        #                 left_rank, right_rank, text_features_to_right)

        #             loss += self._loss(
        #                 image_features,
        #                 text_features_recv,
        #                 logit_scale,
        #                 logit_bias,
        #                 negative_only=True,
        #             )
        #     else:
        #         text_features_to_right = text_features
        #         for i in range(self.world_size - 1):
        #             text_features_from_left = neighbour_exchange_with_grad(
        #                 left_rank, right_rank, text_features_to_right)

        #             loss += self._loss(
        #                 image_features,
        #                 text_features_from_left,
        #                 logit_scale,
        #                 logit_bias,
        #                 negative_only=True,
        #             )
        #             text_features_to_right = text_features_from_left
        return loss if output_dict else loss['contrastive_loss']
        # return {"contrastive_loss": loss, "loss_high_temp": loss_high_temp, "loss_low_temp": loss_low_temp} if output_dict else loss

class BarlowTwinsLoss(nn.Module):
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            lambda_param:float = 0.0051,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.lambda_param = lambda_param

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def _barlowtwins_loss(self, image_features, text_features):
        """
        Implement the barlow twins loss
        """
        image_features = z_score_normalize(image_features) # N, d
        text_features = z_score_normalize(text_features) # N, d
        
        # cross-model correlation matrix mm(z_a_norm.T, z_b_norm) / N
        ccm = image_features.T @ text_features / image_features.shape[0] # d, d

        # loss
        c_diff = (ccm - torch.eye(ccm.shape[0], dtype=ccm.dtype, device=ccm.device)).pow(2)
        # multiply off-diagonal elems of c_diff by lambda
        c_diff = multiply_off_diagonal(c_diff, self.lambda_param)

        loss = c_diff.sum()
        return loss
    
    def forward(self, image_features, text_features, output_dict=False, **kwargs):
        # Normalize features
        loss = self._barlowtwins_loss(image_features, text_features)
        return {"contrastive_loss": loss} if output_dict else loss