import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]) / 256.


class PyrDown(nn.Module):
    r"""Blurs a tensor and downsamples it.

    Args:
        input (torch.Tensor): the tensor to be downsampled.

    Return:
        torch.Tensor: the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self) -> None:
        super(PyrDown, self).__init__()
        self.kernel: torch.Tensor = _get_pyramid_gaussian_kernel()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # blur image
        x_blur: torch.Tensor = F.conv2d(
            input, kernel, padding=2, stride=1, groups=c)

        # reject even rows and columns.
        out: torch.Tensor = x_blur[..., ::2, ::2]
        return out


def pyrdown(input: torch.Tensor) -> torch.Tensor:
    r"""Blur a tensor and downsample it."""
    return PyrDown()(input)


# def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
def _compute_zero_padding(kernel_size):
    """Computes zero padding tuple."""
    padding = [(k - 1) // 2 for k in kernel_size]
    return padding[0], padding[1]


class MaxBlurPool2d(nn.Module):
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size (int): the kernel size for max pooling..

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Returns:
        torch.Tensor: the transformed tensor.

    Examples:
        >>> input = torch.rand(1, 4, 4, 8)
        >>> pool = MaxblurPool2d(kernel_size=3)
        >>> output = pool(input)  # 1x4x2x4
    """

    def __init__(self, kernel_size: int) -> None:
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(self.kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # compute local maxima
        x_max: torch.Tensor = F.max_pool2d(
            input, kernel_size=self.kernel_size,
            padding=self.padding, stride=1)

        # blur and downsample
        x_down: torch.Tensor = pyrdown(x_max)
        return x_down



######################
# functional interface
######################


def max_blur_pool2d(input: torch.Tensor, kernel_size: int) -> torch.Tensor:
    r"""Creates a module that computes pools and blurs and downsample a given
    feature map.

    See :class:`MaxBlurPool2d` for details.
    """
    return MaxBlurPool2d(kernel_size)(input)