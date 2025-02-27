"""
CorrectorAgent module.

This agent corrects the extraction results based on check feedback.
It outputs a corrected extraction result in the required JSON format.
"""

import json
import re
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.prompt import PromptEngine
from typing import Optional


class CorrectorAgent (AgentBase):
    """
    The CorrectorAgent class is responsible for modifying the extraction results
    based on feedback provided during the validation process.
    It returns the corrected extraction result in a specified JSON format.
    """

    def __init__(
            self,
            name: str,
            model_config_name: str,
            use_memory: bool = True,
            memory_config: Optional[dict] = None,
    ) -> None:
        """
        Initializes the CorrectorAgent with the provided configuration.

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
        Parses the JSON data from the model response.

        Args:
            response (ModelResponse): The model response containing the text.

        Returns:
            ModelResponse: The parsed JSON response.
        """
        # Regular expression to extract the JSON data from the response
        pattern = r"```json(.*)```"
        match = re.search(pattern, response.text, re.DOTALL)
        json_text = match.group(1) if match else response.text
        resu = ModelResponse(text=json_text)
        return resu

    def clean_json_string(self, data: str) -> str:
        """
        Cleans the input string to extract a valid JSON array.

        Args:
            data (str): The input string containing the JSON data.

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

    def reply(self, x: dict = None) -> dict:
        """
        Processes the input data based on feedback, modifies the extraction results,
        and returns the corrected JSON result.

        Args:
            x (dict, optional): The input data to be processed and modified.

        Returns:
            dict: The corrected extraction result in JSON format.
        """
        if x is not None:
            self.memory.add(x)

        self.engine = PromptEngine(self.model)

        # Define the base prompt for correction tasks, recommending using Chinese prompt for coding test.
        baseprompt = """
        根据{context}内的“提取结果”按照“待修改部分”的文本和问题进行修改，输出提取结果：
        1- 不允许修改“文本”内容
        2- 根据“待修改部分”的“位置”定位问题，根据“待修改部分”的“问题”修改“提取结果”
        3- 构件、构件编号、构件部位、病害位置、病害，这5个实体是依次的关系，当其中的实体存在时，不可跨越连接，例如，存在构件、构件编号、病害，只可以是//构件-构件位置是-构件编号-存在病害是-病害//
        4- 检查是否满足8个实体//构件编号（例如：1#、13-2#、第一跨）、构件（湿接缝）、构件部位（模板、底板、腹板、翼缘板、台顶、台帽、横梁）、病害位置（例如：距0#台处1.5m，距左侧人行道4m处、锚固区等）、病害、病害数量（例如：3条等）、病害性状描述类别（例如：宽度、长度、面积等）、病害性状数值（例如：3厘米、3.45平方米等）//进行实体识别；
        5- 检查是否满足4个关系//构件位置是（构件到构件编号的关系)、具体部位是（构件编号到构件部位的关系）、病害具体位置是（构件部位到病害位置的关系）、存在病害是（病害位置到病害的关系）//进行关系识别
        6- 请按照给定输出样式，输出格式：```json 
        [{{
        "文本": "3#台处伸缩缝锚固区混凝土纵向裂缝，l=0.3m，W=0.15mm",
        "三元组": ["构件:伸缩缝>构件位置是>构件编号:3#台",
                  "构件编号:3#台>病害具体位置是>病害位置:锚固区混凝土",
                  "病害位置:锚固区混凝土>存在病害是>病害:纵向裂缝"],
        "属性": ["纵向裂缝>长度>0.3m",
                "纵向裂缝>宽度>0.15mm"]
        }}] 
        ```
        7- 完成句子修改,请注意json格式的正确性，多条数据时最外层应该包含中括号
        请注意，请拒绝需要删除原文本中信息的要求，即待提取信息的“文本”是绝对不可以改变的。
        只需要最终返回```json your_modification_here ``` ,不需要给我其他任何内容。
        """

        # For english report. Recommending using Chinese prompt for coding test.
        baseprompt_en = """
        Please modify the extracted results based on the "parts to modify" text and problems in {context}, and output the modified results:
        1- Do not modify the "text" content.
        2- Based on the "parts to modify" "location" and "problem", modify the "extraction results".
        3- Elements, element_numbers, element_parts, defect_locations, and defects are sequential entities; when these entities exist, the relationship cannot skip between them. For example, if there are component, element_number, and defect, the relationship must be in this order: //element_-> element_location is -> element_number -> exists defect_is -> defect//.
        4- Check if the 8 entities //element_number (e.g., 1#, 13-2#, First Span), element (e.g., expansion joint), element_part (e.g., formwork, bottom plate, web, flange, top of the pier, cap of the pier, cross beam), defect_location (e.g., 1.5m from 0# pier, 4m from the left sidewalk, anchorage zone, etc.), defect, defect_quantity (e.g., 3 cracks), defect_category (e.g., width, length, area), defect_value (e.g., 3 cm, 3.45 m²)// are recognized.
        5- Check for the 4 relations //element_location_is (relation between element_and element_number), specific_part_is (relation between element_number and element_part), defect_location is (relation between element_part and defect_location), exists_defect_is (relation between defect_location and defect)//.
        6- Please output the modified results in the following format: ```json 
        [{
            "text": "Longitudinal cracks in anchorage zone concrete of expansion joint at platform 3#, l=0.3m, W=0.15mm",
            "triples": ["component: expansion_joint > element_location_is > element_number: platform_3#",
                        "element_number: platform_3# > defect_location_is > defect_location: anchorage_zone_concrete",
                        "defect_location: anchorage_zone_concrete > exists_defect_is > defect: longitudinal_cracks"],
            "properties ": ["longitudinal_cracks > length > 0.3m",
                            "longitudinal_cracks > width > 0.15mm"]
        }]
        """

        self.engine = PromptEngine(self.model)
        variables = {"context": x.content}
        finalprompt = self.engine.join(
            baseprompt, format_map=variables
        )

        prompt = self.model.format(
            Msg("system", finalprompt, role="system")
            ,
            self.memory
            and self.memory.get_memory()
            or x,
        )

        response = self.model(
            prompt,
            parse_func=self.parse_json,

        ).text

        msg = Msg(self.name, response, role="corrector")

        self.speak(msg)

        if self.memory:
            self.memory.add(msg)

        return msg
