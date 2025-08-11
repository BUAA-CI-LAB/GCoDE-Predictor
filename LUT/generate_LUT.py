import os
import pickle
import torch


def combine_files_into_pkl(directory_name):
    files = [os.path.join(directory_name, "aggregate.txt"),
             os.path.join(directory_name, "combine.txt"),
             os.path.join(directory_name, "knn.txt"),
             os.path.join(directory_name, "pool.txt")]

    data_dict = {}

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                if len(items) == 4:  # For aggregate, knn, pool files
                    operation, in_dim, out_dim, latency = items
                    key = (operation, int(in_dim), int(out_dim))
                elif len(items) == 5:  # For combine file
                    operation, pool, in_dim, out_dim, latency = items
                    key = (operation, pool, int(in_dim), int(out_dim))
                else:
                    print(f"Unexpected format in {file}: {line}")
                    continue
                latency = None if latency == "OOM" else float(latency)
                data_dict[key] = latency

    # Save the combined data to a pickle file
    output_file = os.path.join(directory_name, 'LUT.pkl')
    torch.save(data_dict, output_file)

    print(f"Data saved to {output_file}")

if __name__ == '__main__':
    name = "pi"
    combine_files_into_pkl(name)
    name = "i7"
    combine_files_into_pkl(name)
    name = "tx2"
    combine_files_into_pkl(name)
    name = "1060"
    combine_files_into_pkl(name)