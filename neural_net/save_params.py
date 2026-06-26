import torch

def write_c_array(name, array, f):
    c_name = name.replace(".", "_")

    flat = array.reshape(-1)

    f.write(f"const float {c_name}[{len(flat)}] = {{\n")

    for i, value in enumerate(flat):
        f.write(f"{float(value):.9g}f")

        if i != len(flat) - 1:
            f.write(", ")

        if (i + 1) % 8 == 0:
            f.write("\n")

    f.write("\n};\n\n")


def export_c(model, filename="model_weights.c"):
    model.eval()

    with open(filename, "w") as f:
        f.write("#include <stdint.h>\n\n")

        for name, tensor in model.state_dict().items():
            array = (
                tensor.detach()
                .cpu()
                .to(torch.float32)
                .contiguous()
                .numpy()
            )
            write_c_array(name, array, f)