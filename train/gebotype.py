from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# # TODO CIFAR-10
CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0),
            ('sep_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_5x5', 4)], normal_concat=[2, 3, 4, 5],
    reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 0),
            ('dil_conv_3x3', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=[2, 3, 4, 5])
# # TODO imagenet
imagenet = Genotype(
    normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0),
            ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])



ZPNAS = CIFAR10



