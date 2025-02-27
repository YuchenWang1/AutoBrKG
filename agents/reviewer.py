"""
ReviewerAgent module.

This agent checks the extracted content for logical consistency, language correctness,
and relationship structure. It outputs the review results with modification suggestions.
"""

import json
import re
from typing import Optional
import agentscope
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.prompt import PromptEngine


class ReviewerAgent(AgentBase):
    """
    The ReviewerAgent class is responsible for checking the extracted results.
    It verifies whether the entities and relationships follow the logical structure and linguistic rules.
    """
    def __init__(
            self,
            name: str,
            model_config_name: str,
            use_memory: bool = True,
            memory_config: Optional[dict] = None,
    ) -> None:
        """
        Initializes the ReviewerAgent with the provided configuration.

        Args:
            name (str): The name of the agent.
            model_config_name (str): The model configuration name.
            use_memory (bool): Whether to use memory for the agent.
            memory_config (Optional[dict]): Optional memory configuration.
        """
        super().__init__(
            name=name,
            model_config_name=model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
        )

    def parse_json(self, response: ModelResponse) -> ModelResponse:
        """
        Extracts the JSON content from the response.

        Args:
            response (ModelResponse): The response object containing the raw text.

        Returns:
            ModelResponse: The parsed JSON content.
        """
        # Regular expression to extract JSON data from the response text
        pattern = r"```json(.*)```"
        match = re.search(pattern, response.text, re.DOTALL)
        json_text = match.group(1) if match else response.text
        resu = ModelResponse(text=json_text)
        return resu

    def clean_json_string(self, data: str) -> str:
        """
        Cleans the input string to extract a valid JSON array.

        Args:
            data (str): The input string containing the JSON content.

        Returns:
            str: The cleaned JSON array string.

        Raises:
            ValueError: If no valid JSON array is found.
        """
        # Use regular expression to find and clean the JSON array
        match = re.search(r'\[.*\]', data, re.DOTALL)
        if match:
            return match.group(0)
        else:
            raise ValueError("No valid JSON array found in the string")

    def save_results_to_json(self, results, file_path="check_results.json"):
        """
        Saves the results into a JSON file.

        Args:
            results (str): The results data in string format.
            file_path (str): The file path to save the results to. Default is "check_results.json".
        """
        data = self.clean_json_string(results)
        data = json.loads(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results have been saved to {file_path}")

    def reply(self, x: dict = None) -> dict:
        """
        Processes the input data, evaluates the extracted content, and returns the review results.

        Args:
            x (dict, optional): The input data to be processed and reviewed.

        Returns:
            dict: The review result message containing suggestions for modification.
        """
        if x is not None:
            self.memory.add(x)

        self.engine = PromptEngine(self.model)

        # Define the base prompt for review tasks, recommending using Chinese prompt for coding test.
        baseprompt = """
                        请检查以下内容{context}是否符合链路顺序和语言学逻辑，并根据规则提出修改建议：
                        1. 检查是否存在以下五类实体：构件、构件编号、构件部位、病害位置、病害。若全都存在，链路应为：
                            【构件】- 构件位置是 -【构件编号】- 具体部位是 -【构件部位】- 病害具体位置是 -【病害位置】- 存在病害是 -【病害】。
                        如果实体位置或关系顺序错误，请指出并建议修改。
                        2. 如果缺少某些实体，请参照以下情况调整链路：
                            - 无构件部位：链路为【构件】- 构件位置是 -【构件编号】- 病害具体位置是 -【病害位置】- 存在病害是 -【病害】。
                            - 无构件编号：链路为【构件】- 具体部位是 -【构件部位】- 病害具体位置是 -【病害位置】- 存在病害是 -【病害】。
                            - 无病害位置：链路为【构件】- 构件位置是 -【构件编号】- 具体部位是 -【构件部位】- 存在病害是 -【病害】。
                            - 无构件部位和病害位置：链路为【构件】- 构件位置是 -【构件编号】- 存在病害是 -【病害】。
                            - 无构件编号和病害位置：链路为【构件】- 具体部位是 -【构件部位】- 存在病害是 -【病害】。
                            - 无构件编号和构件部位：链路为【构件】- 病害具体位置是 -【病害位置】- 存在病害是 -【病害】。
                            - 仅存在构件和病害：链路为【构件】- 存在病害是 -【病害】。
                        3. 进行语言学检查：删除中间空格和连字符，确保句子流畅且符合中文表达逻辑。
                        4. 如果存在疑问，请以JSON格式输出：
                        ```json
                        {{
                            "sentence": "",
                            "存在错误": "",
                            "修正结果": ""
                        }}
                        5. 示例1:
                        输入： 构件: 盖梁 - 存在病害是 - 病害: 渗水污染
                        存在错误：无
                        修正结果：无

                        示例2:
                        输入： 构件编号: 第1跨 - 构件位置是 - 构件: 人行 - 具体部位是 - 构件部位: 左侧人行道 - 病害具体位置是 - 病害位置: 近0#台处 - 存在病害是 - 病害: 面砖破损
                        存在错误：构件名称的实体错误
                        修正结果：构件: 人行道
                        
                        示例3:
                        输入： 构件编号: 第5跨 - 构件位置是 - 构件: 人行道 - 具体部位是 - 病害位置: 近0#台处 - 存在病害是 - 病害: 面砖破损
                        存在错误：具体部位是的关系存在错误
                        修正结果：构件: 人行道 - 病害具体位置是 - 病害位置: 近0#台处

                        示43:
                        输入：构件: 盖梁 - 存在病害是 - 病害: 渗水污染  - 病害具体位置是 - 构件: 侧面
                        存在错误：存在病害是和病害具体位置是链路错误
                        修正结果：桥梁名称: 构件: 盖梁 - 病害具体位置是 - 构件: 侧面 - 存在病害是 - 病害: 渗水污染  

                        语言学检查后，请注意json格式的正确性，多条数据时最外层应该包含中括号
                        特别注意，只需要最终返回 ```json your_check_results_here ``` 不需要给我其他任何内容。
                        """

        # For english report. Recommending using Chinese prompt for coding test.
        baseprompt_en = """
        Please check the following content {context} for compliance with the chain link order and linguistic logic, and suggest modifications based on the rules:
        1. Check if the following five types of entities exist: element, element_number, element_part, defect_location, defect. If all of them exist, the link should be:
            【element】-> element_location_is ->【element_number】-> specific_part_is ->【element_part】-> defect_location_is ->【defect_location】-> exists_defect_is ->【defect】.
        If the entity position or relationship order is incorrect, please point it out and suggest a modification.
        2. If certain entities are missing, adjust the link order according to the following situations:
            - No element_part: the link should be 【element】-> element_location_is ->【element_number】-> defect_location_is ->【defect_location】-> exists_defect_is ->【defect】.
            - No element_number: the link should be 【element】-> specific_part_is ->【element_part】-> defect_location_is ->【defect_location】-> exists_defect_is ->【defect】.
            - No defect_location: the link should be 【element】-> element_location_is ->【element_number】-> specific_part_is ->【element_part】-> exists_defect_is ->【defect】.
            - No element_part and defect_location: the link should be 【element】-> element_location_is ->【element_number】-> exists_defect_is ->【defect】.
            - No element_number and defect_location: the link should be 【element】-> specific_part_is ->【element_part】-> exists_defect_is ->【defect】.
            - No element_number and element_part: the link should be 【element】-> defect_location_is ->【defect_location】-> exists_defect_is ->【defect】.
            - Only elementand defect: the link should be 【element】-> exists_defect_is ->【defect】.
        3. Perform linguistic checks: remove intermediate spaces and hyphens, ensuring the sentences are fluent and follow Chinese linguistic logic.
        4. If there are any questions, output in JSON format:
        ```json
        {
            "sentence": "",
            "error_found": "",
            "correction_result": ""
        }
        5. Example 1:
        Input: element: cap_beam - exists_defect_is - defect: water_seepage_pollution
        Error_found: none
        Correction_result: none

        Example 2:
        Input: element_number: First_span - element_location_is - element: sidewalk - specific_part_is - element_part: left_sidewalk - defect_location_is - defect_location: near_0#pier - exists_defect_is - defect: surface_tile_damage
        Error_found: incorrect_entity_name_for_element
        Correction_result: element: sidewalk

        Example 3:
        Input: element_number: Fifth_span - element_location_is - element: sidewalk - specific_part_is - defect_location_is: near_0#pier - exists_defect_is - defect: surface_tile_damage
        Error_found: incorrect_relationship_for_specific_part
        Correction_result: element: sidewalk - defect_location_is - defect_location: near_0#pier

        Example 4:
        Input: element: cap_beam - exists_defect_is - defect: water_seepage_pollution - defect_location_is - element: side
        Error_found: incorrect_relationship_between_exists_defect_is_and_defect_location_is
        Correction_result: element: cap_beam - defect_location_is - element: side - exists_defect_is - defect: water_seepage_pollution

        After linguistic checks, please ensure the json format is correct. For multiple entries, the outermost layer should contain square brackets.
        Please note, only return the final ```json your_check
        """

        self.engine = PromptEngine(self.model)
        variables = {"context": x.content}
        finalprompt = self.engine.join(
            baseprompt, format_map=variables
        )

        prompt = self.model.format(
            Msg("system", finalprompt, role="system"),
            self.memory
            and self.memory.get_memory()
            or x,
        )

        response = self.model(
            prompt,
            parse_func=self.parse_json,

        ).text

        msg = Msg(self.name, response, role="checker")

        self.save_results_to_json(response)
        self.speak(msg)

        if self.memory:
            self.memory.add(msg)

        return msg
