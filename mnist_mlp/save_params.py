import torch


def c_array_name(name):
    return name.replace(".", "_")


def format_float(value):
    return f"{float(value):.9g}f"


def write_1d_array(name, array, f):
    c_name = c_array_name(name)

    f.write(f"const float {c_name}[{array.shape[0]}] = {{\n")

    for i, value in enumerate(array):
        f.write(format_float(value))

        if i != array.shape[0] - 1:
            f.write(", ")

        if (i + 1) % 8 == 0:
            f.write("\n")

    f.write("\n};\n\n")


def write_2d_array(name, array, f):
    c_name = c_array_name(name)
    rows, cols = array.shape

    f.write(f"const float {c_name}[{rows}][{cols}] = {{\n")

    for row_index, row in enumerate(array):
        f.write("    {")

        for col_index, value in enumerate(row):
            f.write(format_float(value))

            if col_index != cols - 1:
                f.write(", ")

        f.write("}")

        if row_index != rows - 1:
            f.write(",")

        f.write("\n")

    f.write("};\n\n")


def write_c_array(name, array, f):
    if array.ndim == 1:
        write_1d_array(name, array, f)
    elif array.ndim == 2:
        write_2d_array(name, array, f)
    else:
        raise ValueError(f"Unsupported tensor shape for {name}: {array.shape}")


def export_c(model, filename="mlp_weights.c"):
    model.eval()

    with open(filename, "w") as f:
        f.write('#include "mlp_weights.h"\n\n')

        for name, tensor in model.state_dict().items():
            array = (
                tensor.detach()
                .cpu()
                .to(torch.float32)
                .contiguous()
                .numpy()
            )
            write_c_array(name, array, f)
