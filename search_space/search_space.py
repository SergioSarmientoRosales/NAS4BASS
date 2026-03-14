import copy
from config import PRIMITIVES, CHANNELS, REPEAT, KERNELS, Genotype
from search_space.encoding import convert


def decode(genome: list[int]) -> Genotype:
    genome = copy.deepcopy(genome)
    channel_idx = genome.pop(0)
    genotype = convert(genome)

    b1 = genotype[0]
    b2 = genotype[1]
    b3 = genotype[2]

    branch1 = [("channels", CHANNELS[channel_idx])]
    branch2 = [("channels", CHANNELS[channel_idx])]
    branch3 = [("channels", CHANNELS[channel_idx])]

    for block in b1:
        for unit in block:
            branch1.append(
                (PRIMITIVES[unit[0]], [KERNELS[unit[1]], KERNELS[unit[1]]], REPEAT[unit[2]])
            )

    for block in b2:
        for unit in block:
            branch2.append(
                (PRIMITIVES[unit[0]], [KERNELS[unit[1]], KERNELS[unit[1]]], REPEAT[unit[2]])
            )

    for block in b3:
        for unit in block:
            branch3.append(
                (PRIMITIVES[unit[0]], [KERNELS[unit[1]], KERNELS[unit[1]]], REPEAT[unit[2]])
            )

    return Genotype(
        Branch1=branch1,
        Branch2=branch2,
        Branch3=branch3,
    )