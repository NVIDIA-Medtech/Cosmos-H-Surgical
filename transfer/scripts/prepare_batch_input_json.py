# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--control_type", type=str, required=True)
    return parser.parse_args()


# Usage: python scripts/prepare_batch_input_json.py --dataset_path assets --save_path assets/depth.jsonl --control_type depth
# Usage: python scripts/prepare_batch_input_json.py --dataset_path assets --save_path assets/seg.jsonl --control_type seg
# Usage: python scripts/prepare_batch_input_json.py --dataset_path assets --save_path assets/edge.jsonl --control_type edge
# Usage: python scripts/prepare_batch_input_json.py --dataset_path assets --save_path assets/vis.jsonl --control_type vis
# Usage: python scripts/prepare_batch_input_json.py --dataset_path assets --save_path assets/multicontrol.jsonl --control_type multicontrol
def main():
    args = parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path
    control_type = args.control_type

    input_folders = glob.glob(os.path.join(dataset_path, "*_example"))
    output_json = []
    for input_folder in input_folders:
        print(input_folder)
        example_folder_name = os.path.basename(input_folder).replace("_example", "")
        control_spec_file = os.path.join(input_folder, control_type, f"{example_folder_name}_{control_type}_spec.json")
        with open(control_spec_file, "r") as f:
            control_spec = json.load(f)
        relative_path = os.path.dirname(control_spec_file).replace(os.path.dirname(save_path) + "/", "")

        control_spec["prompt_path"] = os.path.join(
            input_folder.replace(os.path.dirname(save_path) + "/", ""), os.path.basename(control_spec["prompt_path"])
        )
        control_spec["video_path"] = os.path.join(
            input_folder.replace(os.path.dirname(save_path) + "/", ""), os.path.basename(control_spec["video_path"])
        )

        if control_type == "multicontrol":
            for control_key in ["depth", "edge", "seg", "vis"]:
                if "control_path" in control_spec[control_key]:
                    control_spec[control_key]["control_path"] = os.path.join(
                        relative_path.replace(control_type, control_key),
                        os.path.basename(control_spec[control_key]["control_path"]),
                    )
        else:
            if "control_path" in control_spec[control_type]:
                control_spec[control_type]["control_path"] = os.path.join(
                    relative_path, os.path.basename(control_spec[control_type]["control_path"])
                )
        output_json.append(control_spec)
    print(f"Saved {len(output_json)} items to {save_path}")
    with open(save_path, "w") as f:
        for item in output_json:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
