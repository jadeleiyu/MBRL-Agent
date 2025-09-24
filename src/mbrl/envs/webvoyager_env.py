import time
import os
import re
import logging
import platform
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from .utils_webarena import fetch_browser_info, fetch_page_accessibility_tree,\
                    parse_accessibility_tree, clean_accesibility_tree


# ---------- Prompt templates ----------
WV_SYSTEM_PROMPT = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Accessibility Tree with numerical label representing information about the page, then follow the guidelines and choose one of the following actions:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage.\n6. Google, directly jump to the Google search page. When you can't find information in some websites, try starting over with Google.
7. Answer. This action should only be chosen when all questions in the task have been solved.
Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Google
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits ENTER key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
3) Execute only one action per iteration. \n4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) When a complex Task involves multiple questions or steps, select \"ANSWER\" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER. 
* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

Your reply should strictly follow the format:
Thought: {Your brief thoughts (briefly summarize the info that will help ANSWER)}
Action: {One Action format you choose}
Then the User will provide:
Observation: {Accessibility Tree of a web page}
"""

WV_INIT_USER_PROMPT = """Now given a task: {instruction}  Please interact with {web} and get the answer. \nObservation: please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"""


def driver_config(args):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options


def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


class WebVoyagerEnv:
    def __init__(self, driver_options, task, args):
        self.task = task
        self.task_dir = os.path.join(args.result_dir, 'task{}'.format(task["id"]))
        os.makedirs(self.task_dir, exist_ok=True)
        self.logger = setup_logger(self.task_dir)
        self.logger.info(f'########## TASK{task["id"]} ##########')

        self.window_height = args.window_height
        self.window_width = args.window_width

        self.driver = webdriver.Chrome(options=driver_options)
        self.driver.set_window_size(self.window_width, self.window_height)  # larger height may contain more web information
        self.driver.get(task['web'])
        try:
            self.driver.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        self.driver.execute_script(
            """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};"""
            )
        time.sleep(5)

    
    def get_webarena_accessibility_tree(self):
        browser_info = fetch_browser_info(self.driver)
        ac_tree_raw = fetch_page_accessibility_tree(browser_info, self.driver, current_viewport_only=True)
        ac_tree, obs_nodes_info = parse_accessibility_tree(ac_tree_raw)
        ac_tree = clean_accesibility_tree(ac_tree)
        # if save_file:
        #     with open(save_file + '.json', 'w', encoding='utf-8') as fw:
        #         json.dump(obs_nodes_info, fw, indent=2)
        #     with open(save_file + '.txt', 'w', encoding='utf-8') as fw:
        #         fw.write(ac_tree)

        return ac_tree, obs_nodes_info


    def extract_information(self, text):
        patterns = {
            "click": r"Click \[?(\d+)\]?",
            "type": r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
            # "delete_and_type": r"Delete_and_Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
            "scroll": r"Scroll \[?(\d+|WINDOW)\]?[; ]+\[?(up|down)\]?",
            "wait": r"^Wait",
            "goback": r"^GoBack",
            "google": r"^Google",
            "answer": r"ANSWER[; ]+\[?(.[^\]]*)\]?"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                if key in ["click", "wait", "goback", "google"]:
                    # no content
                    return key, match.groups()
                else:
                    return key, {"number": match.group(1), "content": match.group(2)} if key in ["type", "scroll"] else {"content": match.group(1)}
        return None, None


    def exec_action_click(self, info, web_ele):
        self.driver.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
        web_ele.click()
        time.sleep(3)

    def exec_action_type(self, info, web_ele):
        warn_obs = ""
        type_content = info['content']

        ele_tag_name = web_ele.tag_name.lower()
        ele_type = web_ele.get_attribute("type")
        # outer_html = web_ele.get_attribute("outerHTML")
        if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
            warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
        try:
            # Not always work to delete
            web_ele.clear()
            # Another way to delete
            if platform.system() == 'Darwin':
                web_ele.send_keys(Keys.COMMAND + "a")
            else:
                web_ele.send_keys(Keys.CONTROL + "a")
            web_ele.send_keys(" ")
            web_ele.send_keys(Keys.BACKSPACE)
        except:
            pass

        actions = ActionChains(self.driver)
        actions.click(web_ele).perform()
        actions.pause(1)

        try:
            self.driver.execute_script(
                """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};"""
            )
        except:
            pass

        actions.send_keys(type_content)
        actions.pause(2)

        actions.send_keys(Keys.ENTER)
        actions.perform()
        time.sleep(10)
        return warn_obs


    def exec_action_scroll(self, info, web_eles, obs_info):
        scroll_ele_number = info['number']
        scroll_content = info['content']
        if scroll_ele_number == "WINDOW":
            if scroll_content == 'down':
                self.driver.execute_script(f"window.scrollBy(0, {self.window_height*2//3});")
            else:
                self.driver.execute_script(f"window.scrollBy(0, {-self.window_height*2//3});")
        else:

            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = self.driver.execute_script(
                "return document.elementFromPoint(arguments[0], arguments[1]);", 
                element_box_center[0], 
                element_box_center[1]
            )
            actions = ActionChains(self.driver)
            self.driver.execute_script("arguments[0].focus();", web_ele)
            if scroll_content == 'down':
                actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
            else:
                actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
        time.sleep(3)


    def step(self, action_text, obs_info) -> str:
        pattern = r'Thought:|Action:|Observation:'
        # extract action info
        try:
            assert 'Thought:' in action_text and 'Action:' in action_text
        except AssertionError as e:
            self.logger.error(e)
            fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
            return

        # bot_thought = re.split(pattern, text)[1].strip()
        chosen_action = re.split(pattern, action_text)[2].strip()
        # print(chosen_action)
        action_key, info = self.extract_information(chosen_action)

        fail_obs = ""  # When error execute the action
        pdf_obs = ""  # When download PDF file
        warn_obs = ""  # Type warning
        # execute action
        try:
            window_handle_task = self.driver.current_window_handle
            self.driver.switch_to.window(window_handle_task)

            if action_key == 'click':
                click_ele_number = info[0]
                element_box = obs_info[click_ele_number]['union_bound']
                element_box_center = (element_box[0] + element_box[2] // 2,
                                        element_box[1] + element_box[3] // 2)
                web_ele = self.driver.execute_script(
                    "return document.elementFromPoint(arguments[0], arguments[1]);", 
                    element_box_center[0], 
                    element_box_center[1]
                )

                ele_tag_name = web_ele.tag_name.lower()
                ele_type = web_ele.get_attribute("type")

                self.exec_action_click(info, web_ele)

                if ele_tag_name == 'button' and ele_type == 'submit':
                    time.sleep(10)

            elif action_key == 'wait':
                time.sleep(5)

            elif action_key == 'type':

                type_ele_number = info['number']
                element_box = obs_info[type_ele_number]['union_bound']
                element_box_center = (element_box[0] + element_box[2] // 2,
                                        element_box[1] + element_box[3] // 2)
                web_ele = self.driver.execute_script(
                    "return document.elementFromPoint(arguments[0], arguments[1]);", 
                    element_box_center[0], 
                    element_box_center[1]
                )

                warn_obs = self.exec_action_type(info, web_ele)
                if 'wolfram' in self.task['web']:
                    time.sleep(5)

            elif action_key == 'scroll':
                self.exec_action_scroll(info, None, obs_info)

            elif action_key == 'goback':
                self.driver.back()
                time.sleep(2)

            elif action_key == 'google':
                self.driver.get('https://www.google.com/')
                time.sleep(2)

            elif action_key == 'answer':
                self.logger.info(info['content'])
                self.logger.info('finish!!')
                return

            else:
                raise NotImplementedError
            fail_obs = ""

        except Exception as e:
            self.logger.error('driver error info:')
            self.logger.error(e)
            if 'element click intercepted' not in str(e):
                fail_obs = (
                    "The action you have chosen cannot be exected. "
                    "Please double-check if you have selected the wrong Numerical Label or Action or Action format. "
                    "Then provide the revised Thought and Action."
                )
            else:
                fail_obs = ""
            time.sleep(2)

        return fail_obs

