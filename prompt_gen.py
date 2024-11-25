import json
import json_repair
from client import Client

class PromptGen:
    def __init__(self, client: Client):
        self.client = client
    
    system_prompt = "You are a justifier chatbot designed to generate step-by-step justifications that derived form the Title and Description of a study and then gradually build up to the Desired criteria. Your task is to analyze the title and description of a study and build logical, step-by-step justifications that connect the study’s key elements to the desired criteria. Reference related example studies if they reinforce your justifications. You must assume the desired criteria are correct (as it was already reviewed by specialists) and develop arguments to support them based on the study context and relevant research insights."
    @staticmethod
    def gen_messages(input):
        return [
            {"role": "system", "content": PromptGen.system_prompt},
            {"role": "user", "content": input},
        ]
    @staticmethod
    def related_studies_template(title: str, description: str, criteria: str):
        return f"""Example Title: {title}
    Example Description: {description}
    Example Criteria: {criteria}
    """

    def craft_context_from_studies_documents(self ,related_studies: list[str]):
        json_related_studies = [json.loads(i) for i in related_studies]
        context = ""
        for i in json_related_studies:
            title = i.get('metadata', {}).get('Official_title', "")
            description = i.get('description', "")
            criteria = i.get('criteria', "")
            if title and description:
                context += f"""<STUDY>
    {self.related_studies_template(title, description, criteria)}
    </STUDY>"""
        return context
    @staticmethod
    def user_prompt_template(encoded_related_studies: str, title: str, description: str, desired_criteria: str):
        user_prompt_template = """<EXAMPLE_STUDIES>{encoded_related_studies}</EXAMPLE_STUDIES>

Title: {title}
Description: {description}
Desired criteria: {desired_criteria}

Task Instructions:
1. Derive a step-by-step justification starting from the "Title" and "Description" provided, gradually building up to support the "Desired criteria".
2. Clearly explain the rationale behind each parameter of all criteria, including values, thresholds, and other specific details.
3. Could use example studies (in the <EXAMPLE_STUDIES> section) if they support your justifications, but ensure the reasoning is well-explained and relevant to the study's context.
4. Avoid mentioning that the criteria were already provided, and please do not cite the "Desired criteria" directly in your justification.
5. You should give the justification first before giving out the criteria.

Response Format:
<STEP-BY-STEP-JUSTIFICATION>
Your long step by step detailed logical justification here.
</STEP-BY-STEP-JUSTIFICATION>
"""

        return user_prompt_template.format(encoded_related_studies=encoded_related_studies, title=title, description=description, desired_criteria=desired_criteria)

    
    def get_messages_for_CoT_huggingface(self, encoded_related_studies: str, title: str, description: str, desired_criteria: str):
        return self.gen_messages(self.user_prompt_template(encoded_related_studies, title, description, desired_criteria))
        

    def get_info_for_prompt_gen(self ,study_info: dict):
        metadata = json_repair.loads(study_info.get('metadata'))
        title = metadata.get('Official_title', '') or metadata.get('Brief_Title', '')
        description = study_info.get('data')
        study_id = metadata.get('NCT_ID')
        desired_criteria = study_info.get('criteria')

        # Ensure we have the minimum required information
        if not title or not description or not desired_criteria or not study_id:
            print(f"Skipping study {study_id}: Missing title or description or desired criteria or study id")
            return None

        query = f'{title} [SEP] {description}'
        relevant_studies = self.client.retrieve_relevant_studies(query, study_id)
        encoded_related_studies = self.craft_context_from_studies_documents([i['document'] for i in relevant_studies])
        return encoded_related_studies, title, description, desired_criteria
    
    @staticmethod
    def gen_input(encoded_realted_studies: str, title: str, description: str):
      return f"""<RELATED_STUDIES>
{encoded_realted_studies}
</RELATED_STUDIES>

Target Study Title: {title}
Target Study Description: {description}

Task Instruction:
1. Based on the "Target Study Title" and "Target Study Description" of the target study, please create a Eligibility Criteria for the target study.
2. In <STEP-BY-STEP-JUSTIFICATION> section, please provide a detailed step-by-step logical justification for the Eligibility Criteria you created.
3. Could use example studies (in the <RELATED_STUDIES> section) if they support your justifications, but ensure the reasoning is well-explained and relevant to the study's context
4. Please provide the Eligibility Criteria in the following format (the item within the square brackets [] are the options that you can choose from):
<STEP-BY-STEP-JUSTIFICATION>
Your long step by step detailed logical justification here.
</STEP-BY-STEP-JUSTIFICATION>
<CRITERIA>
#Eligibility Criteria:
Inclusion Criteria:

* Inclusion Criteria 1
* Inclusion Criteria 2
* Inclusion Criteria 3
* ...

Exclusion Criteria:

* Exclusion Criteria 1
* Exclusion Criteria 2
* Exclusion Criteria 3
* ...

##Sex :
[MALE|FEMALE|ALL]
##Ages : 
- Minimum Age : ... Years
- Maximum Age : ... Years
- Age Group (Child: birth-17, Adult: 18-64, Older Adult: 65+) : [ADULT|CHILD|OLDER ADULT] comma separated

##Accepts Healthy Volunteers:
[YES|NO]
</CRITERIA>
"""
    @staticmethod
    def gen_output(justification: str, criteria: str):
        return f"""{justification}
<CRITERIA>
{criteria.replace('INCLUSION CRITERIA', 'Inclusion Criteria').replace('EXCLUSION CRITERIA', 'Exclusion Criteria')}
</CRITERIA>
"""