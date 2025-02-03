# Wonderful Team: Zero-Shot Physical Task Planning with Visual LLMs
This is the official implementation for the paper "Wonderful Team: Zero-Shot Physical Task Planning with Visual LLMs" (TMLR 2025).

[[arXiv]](https://arxiv.org/abs/2407.19094) [[OpenReview]](https://openreview.net/forum?id=udVkqIDYSM) [[website]](https://wonderful-team-robotics.github.io/)

## Installation

```
git clone git@github.com:wonderful-team-robotics/wonderful_team_robotics.git

conda create -n wt_env python=3.10

cd wonderful_team_robotics
pip install -r requirements.txt
```

### VIMABench Installation
Please follow the installation guide on the official [VIMABench](https://github.com/vimalabs/VIMABench?tab=readme-ov-file#installation) repo.

## Usage

To get started with a simple demo assuming you have a depth camera setup and running, take the following steps:
  ```
  python main.py --env <VIMABench task name> \
                 --env_type vima_bench \
                 --run_number <your run number> [optional] \
                 --vlm <your openai model> \
                 --collect_log [optinal] \
  ```

- Please see the VIMABench [Task Suite](https://github.com/vimalabs/VIMABench?tab=readme-ov-file#task-suite) for available task names.

- If `--collect_log` is set, the results will be logged to `wonderful_team_robotics/<env>_<run_number>` and the agent response will be saved to `wonderful_team_robotics/<env>_<run_number>/response_log.txt`. Otherwise the results will be saved to `wonderful_team_robotics/log`.

## Citation
If you find Wonderful Team useful in your research or applications, please consider citing it by the following BibTeX entry.
```
@misc{wang2024wonderfulteamzeroshotphysical,
      title={Wonderful Team: Zero-Shot Physical Task Planning with Visual LLMs}, 
      author={Zidan Wang and Rui Shen and Bradly Stadie},
      year={2024},
      eprint={2407.19094},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.19094}, 
}
```