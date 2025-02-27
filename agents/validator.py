"""
ValidatorAgent module.

This agent evaluates the extraction results for correctness in terms of entity and relationship structure.
It returns a JSON with modifications needed and a score between 0 and 1.
"""

import json
import re
import ast
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.models import ModelResponse
from agentscope.prompt import PromptEngine
from typing import Optional
from knowledge.knowledge_base import KnowledgeBase

class ValidatorAgent (AgentBase):
    """
    The ValidatorAgent class is responsible for evaluating extraction results.
    It checks the correctness of the entity and relationship structure in the extracted data.
    """

    def __init__(
            self,
            name: str,
            model_config_name: str,
            use_memory: bool = True,
            memory_config: Optional[dict] = None,
    ) -> None:
        """
        Initializes the ValidatorAgent with the provided configuration.

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
        Parses the JSON data from the response.

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

    def generate_prompt(self, text):
        """
        Generates a prompt for knowledge retrieval based on the input text.

        Args:
            text (str): The input text for which the prompt will be generated.

        Returns:
            str: The generated prompt based on similar examples from the knowledge base.
        """
        # Evaluate the input text to find similar examples in the knowledge base
        text=ast.literal_eval(text)
        context=text[0]["文本"]
        kb = KnowledgeBase('./knowledge/knowledge_base.json')
        similar_example = kb.search_similar(context)
        if similar_example:
            prompt = json.dumps(similar_example, ensure_ascii=False)
        else:
            prompt = """ """
        return prompt

    def reply(self, x: dict = None) -> dict:
        """
        Processes the input data, performs evaluation, and returns the validation results.

        Args:
            x (dict, optional): The input data to be processed and evaluated.

        Returns:
            dict: The validation result message and the evaluation score.
        """
        if x is not None:
            self.memory.add(x)

        self.engine = PromptEngine(self.model)

        data = json.loads(x.content)
        for item in data:
            content = item["文本"]

        sample_prompt = self.generate_prompt(x["content"])
        print(sample_prompt)

        # Define the base prompt for the validation task, recommending using Chinese prompt for coding test.
        baseprompt = """
        请根据下列的规则，对{context}内容进行分析和评价，输出【待修改部分】和【评分（0-1分之间）】：
                1- 首先，判断示例{sample}是否和待分析内容类似，如类似「直接输出评分为1」，位置和问题为“文本类似”；
                2- 如果示例不类似或不存在示例则进行下面检查：
                    1）- 格式检查：确保格式都是正确的，全部符号均满足json格式，且仅检查“三元组”和“属性”部分
                    2）- 链路检查： 存在a～h的8种链路情况，请先匹配属于哪一种链路：
                            a:「构件」->构件位置是 ->「构件编号」->具体部位是 ->「构件部位」->存在病害是 ->「病害」；
                            b:「构件」->具体部位是 ->「构件部位」->病害具体位置是 ->「病害位置」->存在病害是 ->「病害」；
                            c:「构件」->构件位置是 ->「构件编号」->病害具体位置是 ->「病害位置」->存在病害是 ->「病害」；
                            d:「构件」->构件位置是 ->「构件编号」->存在病害是 ->「病害」；
                            e:「构件」->具体部位是 ->「构件部位」->存在病害是 ->「病害」；
                            f:「构件」->病害具体位置是 ->「病害位置」->存在病害是 ->「病害」；
                            g:「构件」->构件位置是 ->「构件编号」->具体部位是 ->「构件部位」->病害具体位置是 ->「病害位置」->存在病害是 -->「病害」；
                            h:「构件」->存在病害是 ->「病害」。
                           （1）链路的第一个实体一定是「构件」，不允许是其他类型实体
                           （2）链路不允许出现一个主实体连接两个宾实体，例如：//伸缩缝->构件编号->0#台；伸缩缝->存在病害是->沉积物轻微阻塞//，"伸缩缝"连接两个宾实体情况则不允许；
                           （3）链路中不允许出现病害数量、病害性状描述类别、病害性状数值等属性实体
                    3）- 实体检查：除构件和病害实体外，其余实体允许不存在，检查是否是桥梁用语，构件（例如：人行道、湿接缝、梁、墩、桥台）、构件编号（例如：1#、13-2#、第一跨）、构件部位（例如：模板、底板、腹板、翼缘板、台顶、台帽、横梁）、病害位置（例如：距0#台处1.5m，距左侧人行道4m处、锚固区等）、病害、病害数量（例如：3条等）、病害性状描述类别（例如：宽度、长度、面积等）、病害性状数值（例如：3厘米、3.45平方米等）//；
                           检查是否符合逻辑，是否适合作为实体的命名，例如：//内、外、1处//等单独的词不适合作为单独实体出现，应该删除该部分；
                    4）- 属性检查：属性提取应该为病害的//病害数量、病害性状描述类别、病害性状数值//，可以不存在属性，可以存在任意数量的属性；
                    5）- 检查结果：指定错误位置（具体哪个实体/关系），说明错误原因
                    6）- 结构评分：根据上述问题的错误错一项扣0.1分，满分1分，输出评价分数，没有问题输出1分
                3- 输出例子，应该包含文本用于定位，问题用于修改，评分用于评价：
                {{"待修改部分": {{"位置": ["具体需修改的实体或关系"],
                                "问题": ["具体的问题"]}},"评分": **}}
                特别注意，只需要最终返回 ```json your_evaluate_results_here ``` 不需要给我其他任何内容。
                """

        # For english report. Recommending using Chinese prompt for coding test.
        baseprompt_en ="""
        Please analyze and evaluate the content of {context} according to the following rules, and output [parts to modify] and [score (between 0 and 1)]:
        1- First, check if the example {sample} is similar to the content to be analyzed. If similar, directly output a score of 1 with "text similar".
        2- If the example is not similar or does not exist, proceed with the following checks:
            1) Format check: Ensure that the format is correct, all symbols meet json formatting, and only check the "triples" and "properties" parts.
            2) Link check: There are 8 link cases (a to h), first match the appropriate link:
                a: "element" -> element_location_is -> "element_number" -> specific_part_is -> "element_part" -> exists_defect_is -> "defect";
                b: "element" -> specific_part_is -> "element_part" -> defect_location_is -> "defect_location" -> exists_defect_is -> "defect";
                c: "element" -> element_location_is -> "element_number" -> defect_location_is -> "defect_location" -> exists_defect_is -> "defect";
                d: "element" -> element_location_is -> "element_number" -> exists_defect_is -> "defect";
                e: "element" -> specific_part_is -> "element_part" -> exists_defect_is -> "defect";
                f: "element" -> defect_location is -> "defect_location" -> exists_defect_is -> "defect";
                g: "element" -> element_location_is -> "element_number" -> specific_part_is -> "element_part" -> defect_location is -> "defect_location" -> exists_defect_is -> "defect";
                h: "element" -> exists_defect_is -> "defect".

            3) Entity check: Apart from element_and defect_entities, other entities may be absent. Check if they are bridge-related terms such as: element_(e.g., sidewalk, expansion joint, beam, pier, abutment), element_number (e.g., 1#, 13-2#, First Span), element_part (e.g., formwork, bottom plate, web, flange, top of the pier, cap of the pier, cross beam), defect_location (e.g., 1.5m from 0# pier, 4m from the left sidewalk, anchorage zone, etc.), defect, defect_quantity (e.g., 3 cracks), defect_category (e.g., width, length, area), defect_value (e.g., 3 cm, 3.45 m²). Check for logical correctness, and ensure suitable entity names.
            4) Attribute check: Attribute extraction should focus on the defect's //defect_quantity, defect_category, defect_value//. Attributes may be absent or present in any quantity.
            5) Check results: Specify error location (which entity/relationship) and describe the error.
            6) Structure score: Deduct 0.1 points for each error based on the above issues, with a full score of 1. Output evaluation score if no issues, otherwise output a score less than 1.
        3- Output example, which should include text to locate the issue, problem description, and score:
        {
            "parts to modify": {
                "location": ["specific_entity or relationship to modify"],
                "problem": ["specific_problem"]
            },
            "score": **
        }
        Just return the final ```json your_evaluate_results_here``` , no other content required.
        """

        self.engine = PromptEngine(self.model)
        variables = {"context": x.content, "sample": sample_prompt}
        finalprompt = self.engine.join(
            baseprompt, format_map=variables
        )

        prompt = self.model.format(
            Msg("system", finalprompt, role="system"),
            x,
        )

        response = self.model(
            prompt,
            parse_func=self.parse_json,

        ).text
        score = json.loads(response)["评分"]

        msg = Msg(self.name, response, role="validator")

        self.speak(msg)

        return msg, score
