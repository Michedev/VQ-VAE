from typing import Any

import torch
from torch import autograd


class VectorQuantizer(autograd.Function):

    @staticmethod
    def forward(ctx: Any, e: torch.Tensor, w_embedding: torch.Tensor) -> Any:
        """
        Quantize the embedding

        >>> w = torch.tensor([[1, 2, 3], [4, 5, 6]]).float() # 2 x 3
        >>> e = torch.zeros(32, 15, 3).float()
        >>> result = VectorQuantizer.apply(e, w)
        >>> result.shape
        torch.Size([32, 15, 3])
        >>> (result.sum().item() == 6 * 15 * 32)  # i.e. always closer to the first vector [1, 2, 3]
        True
        >>> e = torch.zeros(2, 4, 3)
        >>> e[1] += 5
        >>> result = VectorQuantizer.apply(e, w)
        >>> (result[0].sum().item() == 4 * 6)
        True
        >>> (result[1].sum().item() == 4 * 15)
        True

        @param e: the embedding tensor with shape (batch_size, length, embedding_dim)
        @param w_embedding: the embedding dictionary with shape (codebook_length, embedding_dim)

        @return the quantized embedding with shape (batch_size, length, embedding_dim)
        """
        B = e.shape[0]  # batch size
        E = w_embedding.shape[-1]  # embedding size
        with torch.no_grad():
            dist = torch.cdist(e, w_embedding)
            i_min = torch.argmin(dist, dim=-1)
        result = w_embedding.unsqueeze(0).expand(B, -1, -1).gather(dim=1, index=i_min.unsqueeze(-1).expand(-1, -1, E))
        ctx.save_for_backward(e, w_embedding, i_min, result)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        grad_e = None
        grad_w_embedding = None
        # print('grad_output.shape =', grad_output.shape)
        if ctx.needs_input_grad[0]:
            grad_e = grad_output.clone()
        if ctx.needs_input_grad[1]:
            e, w_embedding, i_min, e_hat = ctx.saved_tensors
            grad_w_embedding = torch.zeros_like(w_embedding)
            embedding_size = grad_output.shape[-1]
            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_w_embedding = grad_w_embedding.index_add(dim=0, index=i_min.view(-1),
                                                          source=grad_output_flatten)
        return grad_e, grad_w_embedding


quantize = VectorQuantizer.apply


def reshape2d_quantize(encoded, codebook):
    """
    Quantize image embedding, by initially flattening the width and height dimensions,
    then apply vector quantization and finally reshaping it to an image.
    @param encoded:
    @param codebook:
    @return: The triplet (flatten embedding, flatten quantized embedding, quantized embedding) with shapes
    (batch_size, length, embedding_dim), (batch_size, length, embedding_dim), (batch_size, embedding_dim, height, width)
    """
    w_e, h_e = encoded.shape[2:]
    embedding_size = codebook.shape[-1]
    e = encoded.flatten(start_dim=2)
    e = e.permute(0, 2, 1)
    e_quantized = quantize(e, codebook)
    e_quantized_reshaped = e_quantized.permute(0, 2, 1)
    e_quantized_reshaped = e_quantized_reshaped.reshape(-1, embedding_size, w_e, h_e)
    return e, e_quantized, e_quantized_reshaped
