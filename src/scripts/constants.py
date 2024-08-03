SEED = 42

UNITS_DICT = {'hemoglobin': 'g/dL', 'ferritin': 'ng/mL', 'ret_count': '%', 'segmented_neutrophils': '', 
              'tibc': 'mcg/dL', 'mcv': 'fL', 'serum_iron': 'mcg/dL', 'rbc': 'millions per cubic litre', 
              'gender': '', 'creatinine': 'mg/dL', 'cholestrol': 'mg/dL', 'copper': 'μg/dL', 'ethanol': '%', 
              'folate': 'ng/mL', 'glucose': 'mg/dL', 'hematocrit': '%', 'tsat': '%'}


CLASS_DICT = {0:'no anemia', 1:'vitamin b12/folate deficiency anemia', 2:'unspecified anemia', 3:'anemia of chronic disease', 
              4:'iron deficiency anemia', 5:'hemolytic anemia', 6:'aplastic anemia', 7:'inconclusive diagnosis'}


ABBRS_DICT = {'mcv': 'mean corpuscular volume', 'rbc':'red blood cell count', 'tibc': 'total iron binding capacity',
             'tsat': 'transferrin saturation', 'ret_count': 'reticulocyte count', 'serum_iron': 'serum iron',
             'segmented_neutrophils':'segmented neutrophils'}

APPROVED_FEATURES = ['hemoglobin', 'gender', 'mean corpuscular volume', 'ferritin', 'total iron binding capacity',
                    'reticulocyte count', 'segmented neutrophils', 'tibc']

TEST_SET_PATH = "../../data/test_set_constant.csv"

map1 = [
    'no anemia',
    'vitamin b12/folate deficiency anemia',
    'unspecified anemia',
    'anemia of chronic disease',
    'iron deficiency anemia',
    'hemolytic anemia',
    'aplastic anemia',
    'inconclusive diagnosis',
    'hemoglobin',
    'ferritin',
    'ret_count',
    'segmented_neutrophils',
    'tibc',
    'mcv',
    'gender',
    'serum_iron',
    'rbc',
    'creatinine',
    'cholestrol',
    'copper',
    'ethanol',
    'folate',
    'glucose',
    'hematocrit',
    'tsat',
    'none'
]

map2 = [
    'no anemia',
    'vitamin b12/folate deficiency anemia',
    'unspecified anemia',
    'anemia of chronic disease',
    'iron deficiency anemia',
    'hemolytic anemia',
    'aplastic anemia',
    'inconclusive diagnosis',
    'hemoglobin',
    'ferritin',
    'reticulocyte count',
    'segmented neutrophils',
    'total iron binding capacity',
    'mean corpuscular volume',
    'gender',
    'none'
]

base_prompt = """
I am going to provide patient information including their gender and some laboratory test 
results. In return, I would like to know if the patient has anemia or not. If they do, I would like to know what kind of
anemia they have. Keep in mind this possible set of anemia types: No anemia, Vitamin B12/Folate deficiency anemia, Unspecified anemia, Anemia of chronic disease, 
Iron deficiency anemia, Hemolytic anemia, Aplastic anemia, Inconclusive diagnosis. If they do not have anemia, the diagnosis should be No anemia. 
If you are not sure, the diagnosis should be Inconclusive diagnosis. Your response should be just the diagnosis and nothing else. No explanation is 
required. 
"""

base_1shot_prompt = """
I am going to provide patient information including their gender and some laboratory test 
results. In return, I would like to know if the patient has anemia or not. If they do, I would like to know what kind of
anemia they have. Keep in mind this possible set of anemia types: No anemia, Vitamin B12/Folate deficiency anemia, Unspecified anemia, Anemia of chronic disease, 
Iron deficiency anemia, Hemolytic anemia, Aplastic anemia, Inconclusive diagnosis. If they do not have anemia, the diagnosis should be No anemia. 
If you are not sure, the diagnosis should be Inconclusive diagnosis. Your response should be just the diagnosis and nothing else. No explanation is 
required. For example, you have that hemoglobin: 10g/dL, mean corpuscular volume: 83fL, reticulocyte count:1.6%, The diagnosis will be Aplastic anemia.
"""

plainCOT_prompt = """
You are a clinician who is skilled in assessing whether a patient has anemia or not, and what type of
anemia they have. You make the diagnosis based on their gender and laboratory test results.
In your clinician role, you will give me the name of every
lab test you took into consideration to determine the final diagnosis and the final diagnosis at the end. If you reached the conclusion of Inconclusive Diagnosis please don't try to look for other solutions.
Usually you make a diagnosis based on the following rules:
1) look for the hemoglobin value first. 2) If the hemoglobin value is greater than 13 g/dL, the diagnosis is No anemia. If the hemoglobin value is greater
than 12 g/dL but less than 13g/dL, look for the gender of the patient.
3) If the patient is female and their hemoglobin value is greater than 12g/dL, the diagnosis is No anemia.
4) Otherwise, look for the mean corpuscular volume. Here you can distinghuish four cases (named a,b,c,d):
case a) If the mean corpuscular volume results are unavailable, the diagnosis is Inconclusive diagnosis.
case b)If the mean corpuscular volume is less than 80fL, look for the ferritin value. If the
ferritin results are unavailable, the diagnosis is Inconclusive diagnosis. If the ferritin value is less than 30ng/ml, the diagnosis is Iron
deficiency anemia. If the ferritin value is greater than 100ng/ml, the
diagnosis is Anemia of chronic disease. If the ferritin value is between 30ng/ml and 100ng/ml, you will look for the tibc.
If the tibc is less than 450mcg/dL, the diagnosis is Anemia of chronic disease. If the tibc
results are unavailable, the diagnosis is Inconclusive diagnosis. If the tibc result
is higher than 450mcg/dL, the diagnosis is Iron deficiency anemia.
case c) If the mean corpuscular volume is greater than 100fL, look for the value of: segmented neutrophils.
If the segmented neutrophils results are unavailable, the diagnosis is Inconclusive diagnosis. If the segmented
neutrophils value is equal to 0, the diagnosis is Unspecified anemia. If the segmented neutrophils have a value greater
than 0, the diagnosis is Vitamin B12/Folate deficiency anemia.
case d) If the mean corpuscular volume is between 80fL and 100fL, look for the reticulocyte count.
If the reticulocyte count results are unavailable, the diagnosis is Inconclusive
diagnosis. If the reticulocyte count is less or equal to 2%, the diagnosis is Aplastic anemia. If the reticulocyte count is greater than 2%,
the diagnosis is Hemolytic anemia. Your final response should be the diagnosis and nothing else.
For example, you have that hemoglobin: 10g/dL, mean corpuscular volume: 83fL, reticulocyte count:1.6%, The diagnosis will be Aplastic anemia.
"""

plainNOCOT_prompt = """
I am going to provide patient information including their gender and some laboratory test 
results. In return, I would like to know if the patient has anemia or not. If they do, I would like to know what kind of
anemia they have. Keep in mind this possible set of anemia types: No anemia, Vitamin B12/Folate deficiency anemia, Unspecified anemia, Anemia of chronic disease, 
Iron deficiency anemia, Hemolytic anemia, Aplastic anemia, Inconclusive diagnosis. If they do not have anemia, the diagnosis should be No anemia. 
If you are not sure, the diagnosis should be Inconclusive diagnosis.
Usually you make a diagnosis based on the following rules:
1) look for the hemoglobin value first. 2) If the hemoglobin value is greater than 13 g/dL, the diagnosis is No anemia. If the hemoglobin value is greater
than 12 g/dL but less than 13g/dL, look for the gender of the patient.
3) If the patient is female and their hemoglobin value is greater than 12g/dL, the diagnosis is No anemia.
4) Otherwise, look for the mean corpuscular volume. Here you can distinghuish four cases (named a,b,c,d):
case a) If the mean corpuscular volume results are unavailable, the diagnosis is Inconclusive diagnosis.
case b)If the mean corpuscular volume is less than 80fL, look for the ferritin value. If the
ferritin results are unavailable, the diagnosis is Inconclusive diagnosis. If the ferritin value is less than 30ng/ml, the diagnosis is Iron
deficiency anemia. If the ferritin value is greater than 100ng/ml, the
diagnosis is Anemia of chronic disease. If the ferritin value is between 30ng/ml and 100ng/ml, you will look for the tibc.
If the tibc is less than 450mcg/dL, the diagnosis is Anemia of chronic disease. If the tibc
results are unavailable, the diagnosis is Inconclusive diagnosis. If the tibc result
is higher than 450mcg/dL, the diagnosis is Iron deficiency anemia.
case c) If the mean corpuscular volume is greater than 100fL, look for the value of: segmented neutrophils.
If the segmented neutrophils results are unavailable, the diagnosis is Inconclusive diagnosis. If the segmented
neutrophils value is equal to 0, the diagnosis is Unspecified anemia. If the segmented neutrophils have a value greater
than 0, the diagnosis is Vitamin B12/Folate deficiency anemia.
case d) If the mean corpuscular volume is between 80fL and 100fL, look for the reticulocyte count.
If the reticulocyte count results are unavailable, the diagnosis is Inconclusive
diagnosis. If the reticulocyte count is less or equal to 2%, the diagnosis is Aplastic anemia. If the reticulocyte count is greater than 2%,
the diagnosis is Hemolytic anemia. Your final response should be the diagnosis and nothing else.
For example, you have that hemoglobin: 10g/dL, mean corpuscular volume: 83fL, reticulocyte count:1.6%, The diagnosis will be Aplastic anemia.
Your response should be just the diagnosis and nothing else. No explanation is 
required.
"""

sequentialCOT_prompt = """
You are a clinician skilled in diagnosing anemia and determining its type based on a patient's gender and laboratory test results. 
You will request one feature at a time and then provide a final diagnosis. 
At each step, you will explain your reasoning before asking for the next feature and before giving the final diagnosis.
Your reasoning should follow these rules:
1) Ask for the hemoglobin value first.
2) If the hemoglobin value is greater than 13 g/dL, the diagnosis is No anemia. Otherwise if the hemoglobin value is greater
than 12 g/dL but less than 13 g/dL, ask for the gender of the patient.
3) If the patient is female and their hemoglobin value is greater than 12 g/dL, the diagnosis is No anemia, even if the hemoglobin value is not greater than 13 g/dL.
Otherwise ask for the mean corposcular volume.
4) For the mean corposcular volume you can distinguish four cases (named a,b,c,d):
case a) If the mean corpuscular volume results are unavailable, the diagnosis is Inconclusive diagnosis.
case b) If the mean corpuscular volume is less than 80 fL, ask for the ferritin value. If the
ferritin results are unavailable, the diagnosis is Inconclusive diagnosis. If the ferritin value is less than 30 ng/ml, the diagnosis is Iron
deficiency anemia. If the ferritin value is greater than 100 ng/ml, the
diagnosis is Anemia of chronic disease. If the ferritin value is between 30 ng/ml and 100 ng/ml, ask for the tibc.
If the tibc is less than 450 mcg/dL, the diagnosis is Anemia of chronic disease. If the tibc
results are unavailable, the diagnosis is Inconclusive diagnosis. If the tibc
is more than 450 mcg/dL, the diagnosis is Iron deficiency anemia.
case c) If the mean corpuscular volume is greater than 100 fL, ask for the value for segmented neutrophils.
If the segmented neutrophils results are unavailable, the diagnosis is Inconclusive diagnosis. If the segmented
neutrophils have a value of 0, the diagnosis is Unspecified anemia. If the segmented neutrophils have a value greater
than 0, the diagnosis is Vitamin B12/Folate deficiency anemia.
case d) If the mean corpuscular volume is between 80 fL and 100 fL, ask for the reticulocyte count. If the reticulocyte count results are unavailable, the diagnosis is Inconclusive
diagnosis. If the reticulocyte count is less or equal to 2%, the diagnosis is Aplastic anemia. If the reticulocyte count is greater than 2%,
the diagnosis is Hemolytic anemia. In asking questions please rely solely on the rules.
Please ask for the next laboratory test or gender at the end of your chain of thought not before.
For example, the conversation could go like this: You: ask for hemoglobin, reponse: 10 g/dL, You: ask for the mean corpuscular volume, response: 83 fL,
You:  ask for reticulocyte count, Response:1.6%, You: "diagnosis found: Aplastic anemia".
Remember to use the phrase "diagnosis found" when you reach the final diagnosis.
""".replace('\n', '')

sequentialNOCOT_prompt = """
You are a clinician who is skilled in assessing whether a patient has anemia or not, and what type of
anemia they have. You make the diagnosis based on their gender and laboratory test results. In your clinician role, you ask for a single
feature at each step and finally give a diagnosis. Your request should
only be the name of the feature whose value you want and no other text. You make a diagnosis based on the following rules: 
1) Ask for the hemoglobin value first. 2) If the hemoglobin value is greater than 13 g/dL, the diagnosis is No anemia. If the hemoglobin value is greater
than 12 g/dL but less than 13g/dL, ask for the gender of the patient. 
3) If the patient is female and their hemoglobin value is greater than 12g/dL, the diagnosis is No anemia. 
4) Otherwise, ask for the mean corpuscular volume. Here you can distinghuish four cases (named a,b,c,d):
case a) If the mean corpuscular volume results are unavailable, the diagnosis is Inconclusive diagnosis. 
case b) If the mean corpuscular volume is less than 80fL, ask for the ferritin value. If the
ferritin results are unavailable, the diagnosis is Inconclusive diagnosis. If the ferritin value is less than 30ng/ml, the diagnosis is Iron
deficiency anemia. If the ferritin value is greater than 100ng/ml, the
diagnosis is Anemia of chronic disease. If the ferritin value is between 30ng/ml and 100ng/ml, ask for the tibc.
If the tibc is less than 450mcg/dL, the diagnosis is Anemia of chronic disease. If the tibc
results are unavailable, the diagnosis is Inconclusive diagnosis. If the tibc
is more than 450mcg/dL, the diagnosis is Iron deficiency anemia. 
case c) If the mean corpuscular volume is greater than 100fL, ask for the value for segmented neutrophils. 
If the segmented neutrophils results are unavailable, the diagnosis is Inconclusive diagnosis. If the segmented
neutrophils have a value of 0, the diagnosis is Unspecified anemia. If the segmented neutrophils have a value greater
than 0, the diagnosis is Vitamin B12/Folate deficiency anemia.
case d) If the mean corpuscular volume is between 80fL and 100fL, ask for the reticulocyte count. If the reticulocyte count results are unavailable, the diagnosis is Inconclusive
diagnosis. If the reticulocyte count is less or equal to 2%, the diagnosis is Aplastic anemia. If the reticulocyte count is greater than 2%,
the diagnosis is Hemolytic anemia. Your final response should be the diagnosis and nothing else. For example, the conversation could go like
this: You: hemoglobin, reponse: 10g/dL, You: mean corpuscular volume, response: 83fL, You: reticulocyte count, Response:1.6%, You: Aplastic
anemia. Please write the final diagnosis in the exactly same way it is written in the rules above.
""".replace('\n', '')