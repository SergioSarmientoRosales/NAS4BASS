from typing import List, Union


def gray_to_int(gray_code: Union[str, list[int]]) -> int:
    if isinstance(gray_code, list):
        gray_bits = [int(bit) for bit in gray_code]
    else:
        gray_bits = [int(bit) for bit in str(gray_code)]

    binary_bits = [gray_bits[0]]

    for i in range(1, len(gray_bits)):
        next_binary_bit = gray_bits[i] ^ binary_bits[i - 1]
        binary_bits.append(next_binary_bit)

    binary_str = "".join(str(bit) for bit in binary_bits)
    return int(binary_str, 2)


def bstr_to_rstr(bstring: Union[list[int], str]) -> List[int]:
    if isinstance(bstring, list):
        bstring = "".join(str(x) for x in bstring)

    rstr = []
    for i in range(0, len(bstring), 3):
        r = gray_to_int(bstring[i:i + 3])
        rstr.append(r)
    return rstr


def convert_cell(cell_bit_string: list[int]) -> list:
    tmp = [cell_bit_string[i:i + 3] for i in range(0, len(cell_bit_string), 3)]
    return [tmp[i:i + 3] for i in range(0, len(tmp), 3)]


def convert(bit_string: list[int]) -> list:
    third = len(bit_string) // 3
    b1 = convert_cell(bit_string[:third])
    b2 = convert_cell(bit_string[third: third * 2])
    b3 = convert_cell(bit_string[third * 2:])
    return [b1, b2, b3]