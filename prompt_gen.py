import json
import json_repair
from client import Client

class PromptGen:
    def __init__(self, client: Client):
        self.client = client
    
    system_prompt = "You are a derive chatbot designed to generate step-by-step deriviation that gradually derived the Desired criteria form the Title and Description of a study. Your task is to analyze the title and description of a study and build logical, step-by-step deriviation that connect the study’s key elements to the desired criteria. Reference related example studies if they reinforce your justifications. You must assume the desired criteria are correct (as it was already reviewed by specialists) and develop arguments to support them based on the study context and relevant research insights."
    @staticmethod
    def gen_messages(input):
        return [
            {"role": "system", "content": PromptGen.system_prompt},
            {"role": "user", "content": input},
        ]
    @staticmethod
    def related_studies_template(nct_id: str, title: str, description: str, criteria: str):
        return f"""Related NCT_ID: {nct_id}
    Related Title: {title}
    Related Description: {description}
    Related Criteria: {criteria}
    """

    def craft_context_from_studies_documents(self ,related_studies: list[str]):
        json_related_studies = [json.loads(i) for i in related_studies]
        context = ""
        for i in json_related_studies:
            title = i.get('metadata', {}).get('Official_title', "")
            description = i.get('description', "")
            criteria = i.get('criteria', "")
            nct_id = i.get('metadata', {}).get('NCT_ID', "")
            if title and description:
                context += f"""<STUDY>
    {self.related_studies_template(nct_id, title, description, criteria)}
    </STUDY>"""
        return context
    @staticmethod
    def user_prompt_template(encoded_related_studies: str, title: str, description: str, desired_criteria: str):
        user_prompt_template = """<RELATED_STUDIES>{encoded_related_studies}</RELATED_STUDIES>

Title: {title}
Description: {description}
Desired criteria: {desired_criteria}

Task Instructions:
1. Derive a step-by-step deriviation starting from the "Title" and "Description" provided, gradually building up to support the "Desired criteria".
2. Clearly explain the rationale behind each parameter of all criteria, including values, thresholds, and other specific details.
3. Could use example studies (in the <RELATED_STUDIES> section) if they support your justifications, but ensure the reasoning is well-explained and relevant to the study's context.
4. Avoid mentioning that the criteria were already provided, and please do not cite the "Desired criteria" directly in your justification.
5. You should give the justification/deriviation first before giving out any thing about the specific criteria/values/parameters. 
    5.1) 
    - BAD EXAMPLE: The study requires a platelet count of >50,000, which is a reasonable threshold to ensure that patients are not at risk of bleeding complications.
    - GOOD EXAMPLE: AS the study aims to investigate post-thrombotic syndrome, it is important to ensure that patients are not at risk of bleeding complications. As also seen in NCT00216866, A platelet count of >50,000 is a reasonable threshold to ensure this.
    5.2)
    - BAD EXAMPLE: The study requires participants with an ejection fraction of <40%, as this ensures reliable outcomes in the population of interest.
    - GOOD EXAMPLE: To evaluate the efficacy of the intervention on heart failure, it is essential to ensure that participants have significant but stable cardiac impairment. This prevents confounding by acute conditions and ensures reliable outcomes. Studies like NCT03536880 set an ejection fraction threshold of <40% for this reason, reflecting patients with systolic dysfunction while avoiding excessively low values that could result in high mortality unrelated to the intervention.

Response Format:
<STEP-BY-STEP-DERIVIATION>
Your long step by step detailed logical deriviation here.
</STEP-BY-STEP-DERIVIATION>
"""

        return user_prompt_template.format(encoded_related_studies=encoded_related_studies, title=title, description=description, desired_criteria=desired_criteria)

    
    def get_messages_for_CoT_huggingface(self, encoded_related_studies: str, title: str, description: str, desired_criteria: str):
        return self.gen_messages(self.user_prompt_template(encoded_related_studies, title, description, desired_criteria))
        

    def get_info_for_prompt_gen(self ,study_info: dict):
        metadata = json_repair.loads(study_info.get('metadata'))
        try:
            title = metadata.get('Official_title', '') or metadata.get('Brief_Title', '')
        except:
            return None
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