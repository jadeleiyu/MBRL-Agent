# ==============================================
# File: collect_real_trajs.py
# ==============================================
import json

from mbrl.envs.webvoyager_env import WV_SYSTEM_PROMPT, WV_INIT_USER_PROMPT, WebVoyagerEnv, driver_config
from mbrl import VanillaPolicy

def main(args):

    # load the policy model
    policy = VanillaPolicy(args)

    # create web driver options
    driver_options = driver_config(args)

    # Load webvoyager tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))


    for task_id in range(len(tasks)):

        task = tasks[task_id]
        env = WebVoyagerEnv(driver_options, task, args)

        ac_tree, obs_info = env.get_webarena_accessibility_tree()
        messages = [
            {"role": "system", "content": WV_SYSTEM_PROMPT},
            {"role": "user", "content": WV_INIT_USER_PROMPT.format(
                instruction=task['instruction'], web=task['web'], ac_tree=ac_tree
            )},
        ]

        it = 0
        while it < args.max_iter:

            ac_tree, obs_info = env.get_webarena_accessibility_tree()
            if it > 0:
                messages.append(
                        {
                        "role": "user", 
                        "content": f"Please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
                    }
                )
            action_text = policy.act(messages)
            messages.append({'role': 'assistant', 'content': action_text})

            env.step(action_text, obs_info)


