# Wonderful Team
<!-- TODO write a project summary -->

## Getting Started
Install any missing packages with ```pip```.
```
pip install -r requirements.txt
```

To get started with a simple demo assuming you have a depth camera setup and running, take the following steps:
1. Modify the ```text_prompt``` at line 18 in ```main.py``` to give your custom instructions.
2. Run the following command to save results in ```./your_task_name```.
  ```
  python main.py --env <your_task_name> --run_number 1 --vlm gpt-4o --collect_log
  ```


<!-- TODO replace paper link -->
For more details, we encourage you to take a look at our [paper](https://wonderful-team-robotics.github.io/media/Wonderful_Team_paper.pdf)
