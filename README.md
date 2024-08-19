# Prompting Large Language Models for Supporting the Differential Diagnosis of Anemia
Clinical diagnosis is theoretically reached by clinicians by following a sequence of steps, such as laboratory exams, observations, or imaging. Diagnosis guidelines are documents authored by expert organizations 
that guide clinicians through these sequences of steps to reach a correct diagnosis. These guidelines are of various quality but have the advantage of following medical reasoning and representing pieces of medical 
knowledge. However, they have two main drawbacks: they are designed to cover the majority of the population, which means they often fail to address patients with uncommon conditions; their update is a long and 
expensive process, making them unadapted to rapidly emerging diseases or to new practices.

Motivated by clinical guidelines, the research team HeKA recently developed a decision support tool with Deep Reinforcement Learning (DRL) algorithms for learning the optimal sequence of actions to perform to 
reach a correct diagnosis. This work presents two main originalities: it relies on patient data only, and it is explanatory, as it provides diagnosis decision pathways, similar to those recommended 
by guidelines. In the context of the emergence of large language models (LLMs), we were curious about how such tools could perform in comparison to diagnosis guidelines and the tool developed by the team. 
To perform such a comparison, during the internship, we considered the clinical use case of the differential diagnosis of anemia. Anemia is a clinical condition defined as a lower-than-normal amount of healthy 
red blood cells in the body, which we chose for three reasons: its diagnosis is mainly based on a series of laboratory tests that are available in most Electronic Health Records (EHRs); 
it is a common diagnosis implying that the associated amount of data may be sufficient to train ML models; and the differential diagnosis of anemia is frequently complex to establish, making its guidance useful.

## Installation
To set up the project, you will need to install the following dependencies:

openai
prefixspan
Levenshtein
mistral7B-v0.3
llama3
ollama
