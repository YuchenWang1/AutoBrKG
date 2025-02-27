"""
DecomposerAgent module.

This agent segments input text into different parts based on given keywords.
It extracts a list of keywords from the input text.
"""

import os
import json
import re
from typing import Optional
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.prompt import PromptEngine


class DecomposerAgent (AgentBase):
    """
    The DecomposerAgent class segments input text based on provided keywords and saves the resulting segmented text into files.
    """

    def __init__(
        self,
        name: str,
        model_config_name: str,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the DecomposerAgent with necessary configurations.

        Args:
            name (str): The name of the agent.
            model_config_name (str): The model configuration to use.
            use_memory (bool): Whether to use memory for the agent.
            memory_config (Optional[dict]): Optional memory configuration.
        """
        super().__init__(
            name=name,
            model_config_name=model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
        )

    def parse_txt(self, response: ModelResponse) -> ModelResponse:
        """
        Parse the response to extract text content.

        Args:
            response (ModelResponse): The model's response containing the text.

        Returns:
            ModelResponse: The parsed response with the text content.
        """
        # Regular expression to extract text from the response
        pattern = r"```txt(.*)```"
        match = re.search(pattern, response.text, re.DOTALL)
        text = match.group(1) if match else response.text
        # res_josn = json.loads(json_text)
        resu=ModelResponse(text=text)
        return resu

    def seg(self, keywords, text):
        """
        Segment the input text based on provided keywords and save the resulting segments into files.

        Args:
            keywords (list): The list of keywords used to segment the text.
            text (str): The input text to be segmented.
        """
        # Create the 'data' folder if it does not exist
        if not os.path.exists('data'):
            os.makedirs('data')

        # Initialize dictionaries to store lines categorized by keywords
        keyword_lines = {keyword: [] for keyword in keywords}
        other_lines = []
        file_count = {keyword: 1 for keyword in keywords}
        other_file_count = 1

        # Segment the text by keywords and classify each line
        for line in text.split("\n"):
            matched = False
            for keyword in keywords:
                if keyword in line:
                    keyword_lines[keyword].append(line.strip() + "\n")
                    matched = True
                    break  # Ensure each line only matches one keyword
            if not matched:
                other_lines.append(line.strip() + "\n")

        # Write the categorized lines into separate files
        for keyword, lines in keyword_lines.items():
            while lines:
                chunk = lines[:1]
                lines = lines[1:]
                filename = f"data/{keyword}-{file_count[keyword]}.txt" if file_count[
                                                                              keyword] > 1 else f"data/{keyword}.txt"
                with open(filename, "w", encoding="utf-8") as file:
                    file.writelines(chunk)
                file_count[keyword] += 1

        # Write the lines that don't match any keyword into other.txt
        while other_lines:
            chunk = other_lines[:1]
            other_lines = other_lines[1:]
            filename = f"data/其他-{other_file_count}.txt" if other_file_count > 1 else "data/其他.txt"
            with open(filename, "w", encoding="utf-8") as file:
                file.writelines(chunk)
            other_file_count += 1

    def reply(self, x: dict = None) -> dict:
        """
        Handle the agent's response to the input data, segment the text based on keywords, and return the result.

        Args:
            x (dict, optional): The input data containing the text to be processed.

        Returns:
            dict: The response message after processing the text.
        """
        if x is not None:
            # Add the input data to memory if provided
            self.memory.add(x)

        # Define the base prompt to classify the text based on structural components, recommending using Chinese prompt for coding test.
        baseprompt = """
        请根据结构构件(例如，桥面铺装、梁、桥台等)，，对{context}内的全部的桥梁检测句子进行分类，输出结构构件名称，注意每个构件名称应该为唯一的构件名称具体输出格式为：
        ```txt
        ["构件名称", "构件名称",...]
        ```
        注意仅输出文本中包含的构件名称，只需要最终返回```txt 全部构件名称 ``` ,不需要给我其他任何内容。
        """

        # For english report. Recommending using Chinese prompt for coding test.
        baseprompt_en = """
        Please classify all the bridge inspection sentences in {context} according to structural elements (such as bridge deck paving, beams, abutments, etc.), and output the names of the structural elements. Note that each element name should be unique. The output format should be as follows:
        ```txt
        ["element name", "element name",...]
        ```
        Note that only the names of the building blocks contained in the text are output, just need to eventually return ``txt all_element_names `` ,don't need to give me anything else.
        """

        # Initialize the prompt engine
        self.engine = PromptEngine(self.model)

        # Define the context variable for the prompt
        variables = {"context": x.content}

        # Join the prompt with the context
        finalprompt = self.engine.join(
            baseprompt, format_map = variables
        )

        # Format the prompt and create a message
        prompt = self.model.format(
            Msg("system", finalprompt, role="system"),
            self.memory
            and self.memory.get_memory()
            or x,
            # type: ignore[arg-type]
        )

        # call llm and generate response
        response = self.model(prompt,
                              parse_func=self.parse_txt,
                              ).text

        msg = Msg(self.name, response, role="decomposer")

        # Create a response message with the generated output
        self.speak(msg)

        # Record the message in memory if available
        if self.memory:
            self.memory.add(msg)

        # Load the keywords from the response and segment the text
        keywords = json.loads(response)
        self.seg(keywords, x.content)

        return msg
