import json
import json_repair

class PromptGen:
    def __init__(self, client):
        self.client = client
    
    system_prompt = "You are a justifier chatbot designed to generate step-by-step justifications derived from the Title and Description of a study and then gradually build up to a specific Desired Criterion. Your task is to analyze the title and description of a study and build logical, step-by-step justifications that connect the study's key elements to the provided criterion. Use in-line references to the given related example studies to reinforce your justifications. Assume the desired criterion is correct (as it has already been reviewed by specialists) and develop arguments to support it based on the study context and relevant research insights."
    @staticmethod
    def gen_messages(input):
        return [
            {"role": "system", "content": PromptGen.system_prompt},
            {"role": "user", "content": input},
        ]
    @staticmethod
    def related_studies_template(nct_id:str ,title: str, description: str, criteria: str):
        return f"""Related NCT_ID: {nct_id}
    Realted Title: {title}
    Realted Description: {description}
    Realted Criteria: {criteria}
    """

    def craft_context_from_studies_documents(self ,related_studies: list[str]):
        json_related_studies = [json.loads(i) for i in related_studies]
        context = ""
        for i in json_related_studies:
            title = i.get('metadata', {}).get('Official_title', "")
            nct_id = i.get('metadata', {}).get('NCT_ID', "")
            description = i.get('description', "")
            criteria = i.get('criteria', "")
            if title and description:
                context += f"""<STUDY>
    {self.related_studies_template(nct_id, title, description, criteria)}
    </STUDY>"""
        return context
    @staticmethod
    def user_prompt_template(encoded_related_studies: str, title: str, description: str, desired_criteria: str):
    # Find number of <STUDY> in encoded_related_studies
        num_studies = encoded_related_studies.count("<STUDY>")
        have_studies = num_studies > 0
        have_studies_structure_section ="""- Include details from related studies if relevant, emphasizing how they support or contrast the criterion.  
  Example:
  - [Related Study Title] excludes/includes [criterion] for [reason].
  - This aligns with the current study goals by [specific justification]."""
        user_prompt_template = (
        f"""{"<RELATED_STUDIES>" + encoded_related_studies + "</RELATED_STUDIES>" if have_studies else ""}

Title: {title}
Description: {description}
Desired criteria: {desired_criteria}

Task Instructions:

Objective:

Provide a concise and logical, step-by-step justification for the specific Desired Criterion based on the study's Title and Description.
Format:

Use the "<JustificationFor>" tag to denote the specific criterion.
Follow the tag with bullet points explaining the rationale behind the criterion.

Structure:
<JustificationFor> [Desired Criterion] </JustificationFor>
- Explain the connection to the study's Title, Description{", or Related Studies" if have_studies else ""}.
  Example:  
  - "The title mentions [specific focus], and the description indicates [details]."  
  {"- Related studies, such as [Study Title] (NCT_ID: [Study ID]), support/exclude this criterion for [reason]." if have_studies else ""}

{have_studies_structure_section if have_studies else ""}
- Provide rationale for how the criterion supports study goals or participant safety.  
  Example:  
  - "This criterion is crucial for [specific goals, e.g., minimizing risks or ensuring ethical compliance]."

- Conclude by summarizing why the criterion is essential.  
  Example:  
  - "Therefore, [criterion] is necessary to achieve [specific outcomes]."

Content Guidelines:
a. Derivation:
    Start by identifying how the study's Title and Description necessitate the Desired Criterion.
    Link specific elements from the Title/Description to the criterion.
b. Rationale:
    Explain why each parameter (values, thresholds, specific details) is set as it is."""
        + (
            "Reference related studies or standards from the provided related studies to support the justification, ensuring to include the NCT_ID."
            if have_studies
            else ""
        )
        + f"""
    Justify all parameters, values, and details mentioned in the criterion.
c. Clarity and Precision:
    Ensure each bullet point is clear, concise, and directly related to the criterion.
    Avoid referencing the "Desired Criterion" explicitly in general terms but ensure focus on the specific eligibility criterion provided.
Final Checks:
    Ensure that all justifications are logically sound, comprehensive, and adhere to ethical and regulatory standards.
    Maintain strict adherence to the prescribed format for consistency and readability.

Example{" " if have_studies else " 1"}:
<JustificationFor> Exclusion of Pregnant or Lactating Women </JustificationFor>
- The description indicates the investigational use of sirolimus, vorinostat, and hydroxychloroquine, all of which carry potential teratogenic risks and the possibility of harm to fetal development. Additionally, lactating women are excluded to prevent exposure of infants to these drugs through breast milk, as the safety of such exposure is not established."""
        + (
            "Related Points from <RELATED_STUDIES>:\n"
            '   - The study "A Phase Ib Study of Rapamycin (Sirolimus) in Patients With Advanced Malignancies" excludes pregnant or lactating women due to the teratogenic and safety concerns associated with rapamycin (NCT_ID: NCT01473940).\n'
            "       - Support: Both studies utilize sirolimus, highlighting consistent teratogenic concerns and the exclusion of these populations for safety reasons.\n"
            '   - The study "A Phase I Study of MLN9708 and Vorinostat to Target Autophagy in Patients With Advanced p53 Mutant Malignancies" similarly excludes pregnant or lactating women to mitigate risks associated with vorinostat (NCT_ID: NCT01475954).\n'
            "       - Support: Vorinostat overlaps as an investigational drug in both studies, underscoring a standard precautionary approach to potential reproductive harm.\n"
            if have_studies and False
            else ""
        )
        + """
- To mitigate these risks, excluding pregnant or lactating women is a standard precaution in clinical trials involving investigational drugs. This requirement aligns with the use of effective contraception throughout the study to prevent unintended pregnancies. The values and thresholds are justified by ethical standards to avoid fetal and infant exposure to drugs with unknown or unsafe reproductive profiles.
Therefore, the exclusion of pregnant or lactating women is necessary to ensure participant safety, prevent potential teratogenic effects, and comply with ethical and regulatory guidelines in clinical research."""
+(
"""
Example 2:
<JustificationFor> Inclusion Criteria: * Creatinine clearance or radioisotope glomerular filtration rate >= 70 mL/min </JustificationFor>
- The study description emphasizes the importance of adequate organ and marrow function, particularly renal function, to ensure patient safety during treatment. A creatinine clearance threshold > 50 mL/min is noted as a critical factor, underlining the need for stringent renal function parameters.
    - Related Points from <RELATED_STUDIES>:
     - The study "A Multi-center Prospective Randomized Phase III Trial to Determine the Benefit of Adjuvant Chemotherapy" (NCT00370890) requires a creatinine clearance > 50 mL/min to ensure renal function sufficient for chemotherapy safety.
        - The higher renal demands of its specific treatment regimen or the need for a broader safety margin may justify a stricter threshold of 70 mL/min, while 50 mL/min ensures basic safety.
- To address treatment-specific risks, higher safety standards are implemented, reducing the likelihood of renal toxicity and necessitating a stricter threshold of 70 mL/min. 
- This criterion ensures patients can tolerate study treatments safely, minimizing risks from chemotherapy and radiotherapy.
Therefore, the inclusion criterion of creatinine clearance or glomerular filtration rate >= 70 mL/min is essential to safeguard patient safety and align with the specific demands of the study protocol."""
)+
f"""
There should be only 1 <JustificationFor></JustificationFor> block. Please only create the justification for {desired_criteria}."""
    )
        return user_prompt_template


    def get_messages_for_CoT_huggingface(self, encoded_related_studies: str, title: str, description: str, desired_criteria: str):
        return self.gen_messages(self.user_prompt_template(encoded_related_studies, title, description, desired_criteria))
        

    def get_info_for_prompt_gen(self ,study_info: dict):
        metadata = json_repair.loads(study_info.get('metadata'))
        title = metadata.get('Official_title', '') or metadata.get('Brief_Title', '')
        description = study_info.get('data')
        study_id = metadata.get('NCT_ID')
        desired_criteria = json.loads(study_info["parsed_ec"][0])

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
3. Could use related studies (in the <RELATED_STUDIES> section) if they support your justifications, but ensure the reasoning is well-explained and relevant to the study's context
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
    @staticmethod
    def _user_prompt_template(encoded_related_studies: str, title: str, description: str, desired_criteria: str):
        user_prompt_template = """<RELATED_STUDIES>{encoded_related_studies}</RELATED_STUDIES>

Title: {title}
Description: {description}
Desired criteria: {desired_criteria}

Task Instructions:

1. Objective:
   - Provide a detailed, step-by-step justification for each eligibility criterion of a clinical study based on its Title and Description.

2. Format:
   - For each eligibility criterion, use the `<JustificationFor>` tag to denote the specific criterion.
   - Follow the tag with bullet points that explain the rationale behind the criterion.

3. Structure:
   <JustificationFor> [Desired Eligibility Criterion] </JustificationFor>
   - [First bullet point explaining the connection to the study's Title or Description. The title mentions that..., The description indicates that..., etc.]
      - [Quote the the related points in the <RELATED_STUDIES>, then end with the corresponding NCT_ID]
         - [Explaination on how it would support your justification]
      ...[More points if needed]
   - [Bullet point providing the rationale for the necessity of this criterion, including justification for every specific values or thresholds.]
   - [Final bullet point summarizing why this criterion is essential. Something like, "Therefore, [Fully justified criterion] is necessary"]

4. Content Guidelines:
   - Derivation:
     - Start by identifying how the study's Title and Description necessitate each eligibility criterion.
     - Link specific elements from the Title/Description to the need for the criterion.
   
   - Rationale:
     - Explain why each parameter (values, thresholds, specific details) is set as it is.
     - Reference relevant pharmacological, safety, and ethical considerations.
     - If applicable, incorporate related studies or standards from the provided related studies to support the justification, also make sure to include the NCT_ID in your response.
     - Make sure to include justifications for all parameters, value, details, etc. mentioned in the eligibility criteria.
   
   - Clarity and Precision:
     - Ensure each bullet point is clear, concise, and directly related to the criterion.
     - Avoid mentioning that the criteria were previously provided or referencing the "Desired criteria" directly.
   
   - Consistency:
     - Maintain a consistent format and level of detail for each criterion's justification.
     - Use professional and objective language appropriate for clinical research contexts.

5. Example:
<JustificationFor> Exclusion of Pregnant or Lactating Women </JustificationFor>
- The description indicates the investigational use of sirolimus, vorinostat, and hydroxychloroquine, all of which carry potential teratogenic risks and the possibility of harm to fetal development. Additionally, lactating women are excluded to prevent exposure of infants to these drugs through breast milk, as the safety of such exposure is not established.
   Related Points from <RELATED_STUDIES>:
      - The study "A Phase Ib Study of Rapamycin (Sirolimus) in Patients With Advanced Malignancies" excludes pregnant or lactating women due to the teratogenic and safety concerns associated with rapamycin (NCT_ID: NCT01473940).
         - Support: Both studies utilize sirolimus, highlighting consistent teratogenic concerns and the exclusion of these populations for safety reasons.
      - The study "A Phase I Study of MLN9708 and Vorinostat to Target Autophagy in Patients With Advanced p53 Mutant Malignancies" similarly excludes pregnant or lactating women to mitigate risks associated with vorinostat (NCT_ID: NCT01475954).
         - Support: Vorinostat overlaps as an investigational drug in both studies, underscoring a standard precautionary approach to potential reproductive harm.
- To mitigate these risks, excluding pregnant or lactating women is a standard precaution in clinical trials involving investigational drugs. This requirement aligns with the use of effective contraception throughout the study to prevent unintended pregnancies. The values and thresholds are justified by ethical standards to avoid fetal and infant exposure to drugs with unknown or unsafe reproductive profiles.
- Therefore, the exclusion of pregnant or lactating women is necessary to ensure participant safety, prevent potential teratogenic effects, and comply with ethical and regulatory guidelines in clinical research.

6. Usage of Realted Studies:
   - Utilize the provided realted studies (in the <RELATED_STUDIES> section) to inform and support justifications where relevant.
   - Ensure that any references to realted studies are indirect and serve to reinforce the rationale without quoting or citing them explicitly.

7. Final Checks:
   - Verify that each `<JustificationFor>` tag accurately reflects an eligibility criterion from the study's Desired Criteria.
   - Ensure that all justifications are logically sound, comprehensive, and adhere to ethical and regulatory standards.
   - Maintain the prescribed format strictly to facilitate consistency and readability.
"""
