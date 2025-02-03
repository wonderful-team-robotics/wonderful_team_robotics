# Wonderful Team: Zero-Shot Physical Task Planning with Visual LLMs
This is the official implementation for the paper "Wonderful Team: Zero-Shot Physical Task Planning with Visual LLMs" (TMLR 2025).

[[paper]](https://wonderful-team-robotics.github.io/media/Wonderful_Team_paper.pdf) [[website]](https://wonderful-team-robotics.github.io/)

## Installation
```
git clone git@github.com:wonderful-team-robotics/wonderful_team_robotics.git

conda create -n wt_env python=3.10

cd wonderful_team_robotics
pip install -r requirements.txt
```

### Set up the camera and robot control

Before running the demo code, please set up a depth camera and a robot, implement a camera and robot control code in `utils.py`, and then modify corresponding code in `main.py` at **lines 20-22, 626, 663**.

If you would like to run the demo without a robot, please use local image and comment out relevant code. (**See lines 29-32**)

Please set the following environment variable to use OpenAI API:
```
export OPENAI_API_KEY=your_openai_key
```

## Usage

To get started with a simple demo assuming you have a depth camera setup and running, take the following steps:
  ```
  python main.py --task <your task name> \
                 --env_type <your environment type> \
                 --run_number <your run number> [optional] \
                 --vlm <your openai model> \
                 --collect_log [optinal] \
                 --prompt <your task prompt>
  ```

- Example prompt: *"Draw a five-pointed star shape."*

- If `--collect_log` is set, the results will be logged to `wonderful_team_robotics/<task>_<run_number>` and the agent response will be saved to `wonderful_team_robotics/<task>_<run_number>/response_log.txt`. Otherwise the results will be saved to `wonderful_team_robotics/log`.

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