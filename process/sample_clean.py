def transform_format(s):
    data, time = s.split(':')
    time = time.replace(' ms', '').strip()
    return f"{data.strip()},{time}"


def remove_duplicates(filepath):
    unique_lines = set()
    lines_to_write = []

    with open(filepath, 'r') as infile:
        for line in infile:
            content_before_second_comma = ','.join(line.split(',', 2)[:2])
            if content_before_second_comma not in unique_lines:
                unique_lines.add(content_before_second_comma)
                lines_to_write.append(line)
    with open(filepath, 'w') as outfile:
        outfile.writelines(lines_to_write)



def convert_file_format(input_filename, out_name):
    with open(input_filename, 'r') as infile, open(out_name, 'a') as outfile:
        for line in infile:
            transformed = transform_format(line)
            outfile.write(transformed + "\n")

dataset = "mn"
device = "pi"
edge = "1060"
band = "40mbps"



input_filename1 = f"{device}_{edge}_{band}_log/{device}_{edge}.txt"
out_name = f"../dataset/{dataset}_{device}_{edge}_{band}.txt"

convert_file_format(input_filename1,out_name)

remove_duplicates(out_name)