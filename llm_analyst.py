from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class LLMAnalyst:
    def __init__(self, api_key=None):
        self.llm = (
            ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", api_key=api_key)
            if api_key
            else None
        )

    def generate_insights(self, data, attrition_rate, top_features, fairness_report):
        """Generate natural language insights using LLM"""
        if not self.llm:
            return "LLM insights disabled (no API key provided)"

        system_message = SystemMessage(
            content="""
        You are an experienced HR data analyst. Provide clear, actionable insights about employee attrition.
        Focus on business implications and recommendations, avoiding technical jargon.
        Highlight any potential biases in the data or model.
        """
        )

        data_summary = f"""
        Dataset contains {len(data)} records with {attrition_rate:.1%} attrition rate.
        Top factors: {top_features.to_dict()}
        Fairness report: {fairness_report}
        """

        human_message = HumanMessage(
            content=f"""
        Analyze this employee attrition data:
        {data_summary}
        Provide:
        1. Key findings
        2. Potential bias concerns
        3. HR recommendations
        4. Data quality issues
        """
        )

        response = self.llm([system_message, human_message])
        return response.content
