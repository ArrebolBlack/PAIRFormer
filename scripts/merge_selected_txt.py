import os

def merge_selected_txt(files_to_merge, output_file, encoding="utf-8"):
    """
    Merge selected txt files (in given order).
    Keep only one header (from the first non-empty file).
    """

    header_saved = False

    with open(output_file, "w", encoding=encoding) as fout:
        for fpath in files_to_merge:
            if not os.path.isfile(fpath):
                print(f"⚠️ File not found: {fpath}")
                continue

            with open(fpath, "r", encoding=encoding) as fin:
                lines = fin.readlines()
                if len(lines) == 0:
                    continue  # skip empty files

                header = lines[0]

                # Write header only once
                if not header_saved:
                    fout.write(header)
                    header_saved = True

                # Write data (skip header)
                fout.writelines(lines[1:])

    print(f"✔ 合并完成，输出文件：{output_file}")


if __name__ == "__main__":
    # === 修改这里：选择你要合并的 txt 文件 ===
    files_to_merge = [
        "data/deepTargetPro/test_split_0.txt",
        # "data/deepTargetPro/test_split_1.txt",
        # "data/deepTargetPro/test_split_2.txt",
        # "data/deepTargetPro/test_split_3.txt",
        # "data/deepTargetPro/test_split_4.txt",
        # "data/deepTargetPro/test_split_5.txt",
        "data/deepTargetPro/test_split_6.txt",
        "data/deepTargetPro/test_split_7.txt",
        "data/deepTargetPro/test_split_8.txt",
        "data/deepTargetPro/test_split_9.txt",
    ]

    merge_selected_txt(files_to_merge, "data/deepTargetPro/deepTargetPro_Test_0,6-9.txt")

    # files_to_merge = [
    #     "data/miRAW_Test0.txt",
    #     # "data/miRAW_Test1.txt",
    #     # "data/miRAW_Test2.txt",
    #     # "data/miRAW_Test3.txt",
    #     # "data/miRAW_Test4.txt",
    #     # "data/miRAW_Test5.txt",
    #     "data/miRAW_Test6.txt",
    #     "data/miRAW_Test7.txt",
    #     "data/miRAW_Test8.txt",
    #     "data/miRAW_Test9.txt",
    # ]

    # merge_selected_txt(files_to_merge, "miRAW_Test_0,6-9.txt")
