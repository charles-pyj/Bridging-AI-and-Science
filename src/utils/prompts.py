
prompt_test = f"""
        You are an expert in Computer Science. Your task is to summarize the following three aspects of the papers given the definitions below.
        Definitions of Aspects
        Scientific problems
        - The scientific problems that the literature used Artificial Intelligence to solve.
        - This could encompass various fields
        Artificial Intelligence methods
        - The exact Artificial Intelligence methods that the author uses to address the scientific problems. Please include how the scientific problem is solved using this AI method.
        - Please include how the data is collected and the results of relevant AI methods if possible.
        Context: 
        - The context of the Scientific problems.

        Notes: 
        - If no Artificial Intelligence methods are used, mark it as "N/A" (not applicable). DO NOT come up with your own interpretation.
        - Each aspect should be summarized in 1-2 sentences in most cases.
        - Please try to use exact words from the title and abstract as much as possible

        Response Format: 
        The response should be in the following format: 
        Scientific problems: ...
        Artificial Intelligence methods: ...
        Context: ...

        This is the paper with the title and abstract: 

        Title: 
        Abstract: 
        """

def get_prompt_revise(title,abstract):
    prompt_revise = f"""## Background 
    You are an expert in both science and artificial intelligence. Given a scientific paper, your task is to extract the following three aspects as listed below.            

    ## Aspects to Extract and Definitions

    * Scientific problems: the main scientific problems to be investigated in this paper. 
    * Artificial Intelligence (AI) methods: the AI methods used to address the scientific problems. 
    * AI Employment: how the AI methods are particularly employed to address the scientific problems. This could refer to details including the correspondence of concepts, datasets, experiment design, and evaluation metrics. 

    ## Scientific Paper to Be Extracted

    Title: {title}
    Abstract: {abstract}

    ## Notes

    * If no Artificial Intelligence methods are used, mark it as "N/A" (not applicable).
    * Please try to extract the original texts from the title and abstract. Do not come up with your own interpretation. 

    ## Response Format

    Please output the extraction results in the JSON format as below:

    {{
    "Scientific problems (in short)": "...",
    "Scientific problems (in detail)": "...",
    "AI methods (in short)": "...",
    "AI Employment": "..."
    }}"""
    return prompt_revise

def get_prompt_revise_usage(title,abstract):
    prompt_revise = f"""## Background
You are an expert in both science and artificial intelligence. Given a scientific paper, your task is to extract the following three aspects as listed below.            

## Aspects to Extract and Definitions

* Scientific problems: the main scientific problems to be investigated in this paper. 
* Artificial Intelligence (AI) methods: the AI methods used to address the scientific problems. 
* AI Usage: how the AI methods are particularly used to address the scientific problems. This could refer to details including the correspondence of concepts, datasets, experiment design, and evaluation metrics. 

## Scientific Paper to Be Extracted

Title: {title}
Abstract: {abstract}

## Notes

* If no AI methods are used, mark it as "N/A" (not applicable).
* Please try to extract the original texts from the title and abstract. Do not come up with your own interpretation. 

## Response Format

Please output the extraction results in the JSON format as below:

{{
'Scientific problems (in short)': '...',
'Scientific problems (in detail)': '...',
'AI methods (in short)': '...',
'AI usage': '...'
}}"""
    return prompt_revise

def get_prompt_revise_definition(title,abstract):
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific paper, your task is to extract the following aspects from the title and abstract: 

* Scientific problem: the main scientific problem to be discussed in this paper. 
* Artificial Intelligence (AI) method: the AI method used to address the scientific problem. 
* AI usage: how the AI method is particularly applied to address the scientific problem. This could be relevant to the input/output, adaptation (e.g., training/fine-tuning, function instantiation), evaluation, outcome, etc.

## Scientific Paper to Be Extracted 
Title: {title}
Abstract: {abstract}

## Notes 
* Please try to extract based on the original texts from the title and abstract. Do not include your own interpretation (except for "AI method (definition)"). 
* If no AI method is used, mark it as "N/A" (not applicable). 

## Response Format 
Please output the extraction results in the JSON format as below: 

{{
"Scientific problem (in short)": "...", 
"Scientific problem (in detail)": "...", 
"AI method (in short)": "...", 
"AI method (definition)": "...",
"AI usage": "This paper ..." 
}}"""
    return prompt_revise

def get_prompt_judge(title ,abstract):
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific paper, your task is to extract the following aspects from the title and abstract: 

* Scientific problem: the main scientific problem to be discussed in this paper. 
* Artificial Intelligence (AI) method: the AI method used to address the scientific problem. 
* AI usage: how the AI method is particularly applied to address the scientific problem. This could be relevant to the input/output, adaptation (e.g., training/fine-tuning, function instantiation), evaluation, outcome, etc.

## Scientific Paper to Be Extracted 
Title: {title}
Abstract: {abstract}

## Notes 
* Please try to extract based on the original texts from the title and abstract. Do not include your own interpretation (except for "AI method (definition)"). 
* If no AI method is used, mark it as "N/A" (not applicable). 

## Response Format 
Please output the extraction results in the JSON format as below: 

{{
"Scientific problem (in short)": "...", 
"Scientific problem (in detail)": "...", 
"AI method (in short)": "...", 
"AI method (definition)": "...",
"AI usage": "This paper ..." 
}}"""
    return prompt_revise

def get_prompt_phase1(title,abstract):
    prompt_phase1 = f"""
    ## Background and Task Description 

    You are an expert in research. Given the title and abstract of a paper, your task is to extract the main problem that the paper is investigating and the main method used to address this problem. 

    Particularly, 

    * Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
    * Problem (definition): The detailed definition of the problem. 
    * Problem discipline: The discipline in which the main problem best fits. 
    * Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
    * Method (definition): The detailed definition of the method. 
    * Usage: How the method is specifically applied to address the problem. 

    ## Scientific Paper to Be Extracted 
    Title: {title}
    Abstract: {abstract}

    ## Response Format 

    Please output the extraction results in the JSON format as below. Fields could be "N/A" if no relevant information can be found in the paper title and abstract. 

    {{
        "Problem (keyword/keyphrase)": "...", 
        "Problem (definition)": "...", 
        "Problem Discipline": "...", 
        "Method (keyword/keyphrase)": "...", 
        "Method (definition)": "...", 
        "Usage": "..." 
    }}
    """
    return prompt_phase1

def get_prompt_phase1_test(title,abstract):
    prompt_phase1 = f"""
   ## Background and Task Description 

You are an expert in research. Given the title and abstract of a paper, your task is to extract the main problem that the paper is investigating and the main method used to address this problem. 

Particularly, 

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Requirements 

* Please do not include any method-specific information in problem extraction. 
* Similarly, please do not include any problem-specific information in method extraction. The extracted method description (keyword/keyphrase and definition) should be generic and can be applied across all application domains. 
* Please do not use abbreviations as keywords/keyphrases. 
* Please output the extraction results in the dictionary format as below. Fields could be "N/A" if no relevant information can be found in the paper title and abstract. 

{{
    "Problem (keyword/keyphrase)": "...", 
    "Problem (definition)": "...", 
    "Problem discipline": "...", 
    "Method (keyword/keyphrase)": "...", 
    "Method (definition)": "...", 
    "Usage": "..." 
}}

## Paper to Extract

Title: {title}

Abstract: {abstract}
    """
    return prompt_phase1

def get_prompt_test(title,abstract):
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific paper, your task is to extract the following aspects from the title and abstract: 

* Scientific problem: the main scientific problem to be discussed in this paper. 
* Artificial Intelligence (AI) method: the AI method used to address the scientific problem. 
* AI usage: how the AI method is particularly applied to address the scientific problem. This could be relevant to the input/output, adaptation (e.g., training/fine-tuning, function instantiation), evaluation, outcome, etc.

## Scientific Paper to Be Extracted 
Title: {title}
Abstract: {abstract}

## Notes 
* Please try to extract based on the original texts from the title and abstract. Do not include your own interpretation (except for "AI method (definition)"). 
* If no AI method is used, mark it as "N/A" (not applicable). 
* Try to first categorize AI methods in short and then provide detailed definition.
* Only output AI method when you are sure that AI methods are used in the papers.
## Response Format 
Please output the extraction results in the JSON format as below: 

{{
"Scientific problem (in short)": "...", 
"Scientific problem (in detail)": "...", 
"AI method (in short)": "...", 
"AI method (definition)": "...",
"AI usage": "This paper ..." 
}}"""
    return prompt_revise

# def get_prompt_AI_prediction_direct_test(sci_short,sci_detail):
#     prompt_revise = f"""## Background and Task Description

# You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a summary of a scientific paper, your task is to recommend potential AI methods that can be used to address the scientific problems in the paper.

# * Scientific problem in short: a short summary of the scientific problem.
# * Scientific problem in detail: a more detailed description of the scientific problems that could include the context, method and results.
# * AI recommendation in short: a few keywords of the potential AI methods that can be used to address the scientific problems.
# * AI recommendation in detail: a more detailed description of how the AI methods you recommend can be applied to the scientific problems, this could include visualization, modeling, explaination, etc.
# ## Scientific Paper to Be Extracted 
# Scientific problem in short: {sci_short}
# Scientific problem in detail: {sci_detail}

# ## Notes 
# * Please try to extract based on the original texts from the description of scientific problems. Do not include your own interpretation (except for "AI recommendation in detail"). 
# * If no potential AI method can be used, mark it as "N/A" (not applicable). 
# * Please respond with specific AI methods instead of high-level AI methods.
# ## Response Format 
# Please output the extraction results in the JSON format as below: 

# {{
# "AI recommendation (in short)": "...", 
# "AI recommendation (in detail)": "...",
# }}"""
#     return prompt_revise

def get_prompt_AI_prediction_direct_test(record):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a summary of a scientific paper, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problems in the paper.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Paper to Be Extracted 
{record}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.

## Response Format 
please output the recommendation it has used as a list. The output should be in the JSON format as below:

[
    {{
        "Method (keyword/keyphrase)": "...",
        "Usage": "..."
    }}, 
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_direct_test(record):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a summary of a scientific paper, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problems in the paper.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Paper to Be Extracted 
{record}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.

## Response Format 
please output the recommendation it has used as a list. The output should be in the JSON format as below:

[
    {{
        "Method (keyword/keyphrase)": "...",
        "Usage": "..."
    }}, 
    ...
]"""
    return prompt_revise


def get_prompt_AI_prediction_direct_test_oneshot(record):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem
Please recommend an AI method to address the below scientific problem for writing an academic paper:

{dict_to_line(record)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output one recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *one* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_direct_test_oneshot(record):
    del record['Problem (keyword/keyphrase)']
    del record['Problem (definition)']
    del record['Problem discipline']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}

## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output one recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *one* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_direct_test_threeshot(record):
    del record['Problem (keyword/keyphrase)']
    del record['Problem (definition)']
    del record['Problem discipline']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}

## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output three recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *three* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_direct_test_fiveshot(record):
    del record['Problem (keyword/keyphrase)']
    del record['Problem (definition)']
    del record['Problem discipline']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}

## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output five recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *five* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_direct_test_tenshot(record):
    del record['Problem (keyword/keyphrase)']
    del record['Problem (definition)']
    del record['Problem discipline']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}

## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output ten recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *ten* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise



def get_prompt_AI_prediction_direct_test_threeshot(record):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem
Please recommend three AI methods to address the below scientific problem for writing an academic paper:

{dict_to_line(record)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output three recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *three* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise






def get_prompt_AI_prediction_direct_test_fiveshot(record):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem
Please recommend five AI methods to address the below scientific problem for writing an academic paper:

{dict_to_line(record)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output five recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *five* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise



def get_prompt_AI_prediction_direct_test_tenshot(record):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem
Please recommend ten AI methods to address the below scientific problem for writing an academic paper:

{dict_to_line(record)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output ten recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *ten* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise



def get_prompt_AI_prediction_few_shot_test_new(record,example_records):
    record_copy = record.copy()
    del record_copy['Method (keyword/keyphrase)']
    del record_copy['Method (definition)']
    del record_copy['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if('Problem' in i or 'Discipline' in i):
                string += i
            else: string += (f"AI {i}")
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem. A few examples of relevant papers with similar scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem

Please recommend an AI method to address the below scientific problem for writing an academic paper:

{dict_to_line(record_copy)}


## Examples of AI usage in similar scientific papers:
{examples}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output one recommendation.
* Note that methods used in the examples might be useful to the recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *one* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_few_shot_test_new_oneshot(record,example_records):
    # del record['Problem (keyword/keyphrase)']
    # del record['Problem (definition)']
    # del record['Problem Discipline']
    # del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record_example):
        string = ""
        for i in record_example.keys():
            #print(i)
            if('Method' in i or 'Discipline' in i):
                string += i
            elif 'Usage' in i:
                string += f"AI {i}"
            elif "Problem" in i: 
                string += (f"Scientific {i}")
            string += ": "
            string += record_example[i]
            #print(record_example[i])
            string += "\n"
        return string
    def get_example(ind,record_example):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record_example)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method. A few examples of how similar AI methods are applied to scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}


## Examples of usage for similar AI methods
{examples}
## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output one recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *one* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "Scientific Problem (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_few_shot_test_new_threeshot(record,example_records):
    # del record['Problem (keyword/keyphrase)']
    # del record['Problem (definition)']
    # del record['Problem discipline']
    # del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record_example):
        string = ""
        for i in record_example.keys():
            #print(i)
            if('Method' in i or 'Discipline' in i):
                string += i
            elif 'Usage' in i:
                string += f"AI {i}"
            elif "Problem" in i: 
                string += (f"Scientific {i}")
            string += ": "
            string += record_example[i]
            #print(record_example[i])
            string += "\n"
        return string
    def get_example(ind,record_example):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record_example)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method. A few examples of how similar AI methods are applied to scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}


## Examples of usage for similar AI methods
{examples}
## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output three recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *three* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "Scientific Problem (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise



def get_prompt_sci_prediction_few_shot_test_new_fiveshot(record,example_records):
    # del record['Problem (keyword/keyphrase)']
    # del record['Problem (definition)']
    # del record['Problem discipline']
    # del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record_example):
        string = ""
        for i in record_example.keys():
            #print(i)
            if('Method' in i or 'Discipline' in i):
                string += i
            elif 'Usage' in i:
                string += f"AI {i}"
            elif "Problem" in i: 
                string += (f"Scientific {i}")
            string += ": "
            string += record_example[i]
            #print(record_example[i])
            string += "\n"
        return string
    def get_example(ind,record_example):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record_example)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method. A few examples of how similar AI methods are applied to scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}


## Examples of usage for similar AI methods
{examples}
## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output five recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *five* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "Scientific Problem (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_sci_prediction_few_shot_test_new_tenshot(record,example_records):
    # del record['Problem (keyword/keyphrase)']
    # del record['Problem (definition)']
    # del record['Problem discipline']
    # del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record_example):
        string = ""
        for i in record_example.keys():
            #print(i)
            if('Method' in i or 'Discipline' in i):
                string += i
            elif 'Usage' in i:
                string += f"AI {i}"
            elif "Problem" in i: 
                string += (f"Scientific {i}")
            string += ": "
            string += record_example[i]
            #print(record_example[i])
            string += "\n"
        return string
    def get_example(ind,record_example):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record_example)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an Artificial Intelligence method and its definition, your task is to recommend potential Scientific Problems that can be solved by the given AI method. A few examples of how similar AI methods are applied to scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Artificial Intelligence Method
Please recommend scientific problems for writing an academic paper that can be addressed by the following Artificial Intelligence method:

{dict_to_line(record)}


## Examples of usage for similar AI methods
{examples}
## Notes 
* If no potential scientific problems can be addressed, mark it as "N/A" (not applicable). 
* Please respond with specific scientific problems instead of high-level scientific concepts.
* Output the keyword/keyphrase of the recommended scientific problems and AI method usage as defined above.
* Exactly output ten recommendation.

## Response Format
Please output the recommendation of scientific problems as a list, which has exactly *ten* element. The output should be in the format of a list as below:

[
    {{
        "Scientific Problem (keyword/keyphrase)": "...",
        "Scientific Problem (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise








def get_prompt_AI_prediction_few_shot_test_new_threeshot(record,example_records):
    record_copy = record.copy()
    # del record_copy['Method (keyword/keyphrase)']
    # del record_copy['Method (definition)']
    # del record_copy['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if('Problem' in i or 'Discipline' in i):
                string += i
            else: string += (f"AI {i}")
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem. A few examples of relevant papers with similar scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem

Please recommend an AI method to address the below scientific problem for writing an academic paper:

{dict_to_line(record_copy)}


## Examples of AI usage in similar scientific papers:
{examples}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output three recommendations.
* Note that methods used in the examples might be useful to the recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *three* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise





def get_prompt_AI_prediction_few_shot_test_new_fiveshot(record,example_records):
    record_copy = record.copy()
    del record_copy['Method (keyword/keyphrase)']
    del record_copy['Method (definition)']
    del record_copy['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if('Problem' in i or 'Discipline' in i):
                string += i
            else: string += (f"AI {i}")
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem. A few examples of relevant papers with similar scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem

Please recommend an AI method to address the below scientific problem for writing an academic paper:

{dict_to_line(record_copy)}


## Examples of AI usage in similar scientific papers:
{examples}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output five recommendations.
* Note that methods used in the examples might be useful to the recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *five* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_few_shot_test_new_tenshot(record,example_records):
    record_copy = record.copy()
    del record_copy['Method (keyword/keyphrase)']
    del record_copy['Method (definition)']
    del record_copy['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if('Problem' in i or 'Discipline' in i):
                string += i
            else: string += (f"AI {i}")
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(example_records):
        examples += get_example(ind+1,example_record)
        examples += "\n"

    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem. A few examples of relevant papers with similar scientific problems will be provided.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem

Please recommend an AI method to address the below scientific problem for writing an academic paper:

{dict_to_line(record_copy)}


## Examples of AI usage in similar scientific papers:
{examples}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output ten recommendations.
* Note that methods used in the examples might be useful to the recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *ten* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Method (definition)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise





def get_prompt_science_in_AI_test(title,abstract):
        prompt_revise = f"""## Background and Task Description

You are an expert in natural science. Given the title and abstract of a paper that leveraged Artificial Intelligence to solve natural scientific problems, your task is to extract the natural scientific problems that the paper tries to solve. The scientific problems extracted must be related to natural science like: chemistry, physics, medicine, biology, etc.

* Scientific problem in short: a few keywords of the natural scientific problem that the paper used AI methods to solve.
* Scientific problem in detail: a more detailed description of the scientific problems that could include the context, method, and results.

## Scientific Paper to Be Extracted 
Title: {title}
Abstract: {abstract}

## Notes 
* If no potential natural scientific problems can be found, mark it as "N/A" (not applicable).
* If the paper's research problem is purely related to AI methods without addressing any natural science problem, mark it as "N/A".

## Response Format 
Please output the extraction results in the format as below: 

{{
"Scientific problem (in short)": "...", 
"Scientific problem (in detail)": "..."
}}"""
        return prompt_revise

def get_prompt_science_in_AI_test_classify(title,abstract):
        prompt_revise = f"""## Background and Task Description

You are an expert in Natural Sciences. Given the title and abstract of a paper, your task is to first detect whether the paper uses Artificial Intelligence to solve natural science problems, where Artificial Intelligence generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. If so, you should extract the scientific problems, the AI methods used and how AI methods are applied to solve scientific problems.The extracted scientific problem should not be just the downstream application of AI methods. Rather, it should serve to solve problems that proves to be difficult with traditional methods.

* Interdisciplinary: whether this paper leveraged AI to solve natural science research problems. Could be True or False.
* Scientific problem in short: a few keywords of the natural scientific problem that the paper used AI methods to solve.
* AI methods in short: a few keywords of the AI method that the paper uses to solve scientific problems.
* AI usage: how the AI method is particularly applied to address the scientific problem. This could be relevant to the input/output, adaptation (e.g., training/fine-tuning, function instantiation), evaluation, outcome, etc.

## Scientific Paper to Be Extracted 
Title: {title}
Abstract: {abstract}

## Notes 
* It is possible that the paper does not solve any natural science problems. In that case mark all fields as "N/A".
* The extracted scientific problem should not be related to Artificial Intelligence, ideally it should be from traditional natural science like chemistry, physics, biology, etc.

## Response Format 
Please output the extraction results in the format as below: 

{{
"Interdisciplinary": "..."
"Scientific problem in short": "...",
"AI methods in short": "...", 
"AI usage": "..."
}}"""
        return prompt_revise

def get_prompt_summary_cluster(scientific_list):
    def get_sience(ind):
        scientific = scientific_list[ind]
        short = scientific.split(":")[0]
        detail = scientific.split(":")[-1].lstrip(" ")
        return  f"""Paper {ind}:\nScientific problems in short: {short}\nScientific problems in detail: {detail}\n
        """
    
    examples = ""
    for i in range(len(scientific_list)):
        examples += get_sience(i)
        examples += "\n"
    #print(examples)
    prompt_revise = f"""## Background and Task Description

You are an expert in science. Given a list of descriptions of scientific problems addressed in scientific papers, you are asked to summarize these scientific problems using less than 3 keywords.

* Scientific problem in short: a few keywords of the scientific problem.
* Scientific problem in detail: a more detailed description of the scientific problems that could include the context, method and results.

## Scientific Paper to Be Extracted 
{examples}

## Notes 
* Please try to be be general about the key words summarization, do not include specific scientific problems.
* Use less than 3 keywords.
## Response Format 
Please output the summarized keywords one by one, separated by comas. Remember that you are only permitted to use less than 3 keywords in total.
"""
    return prompt_revise

    
def get_prompt_phase2(title,abstract,results):
    prompt_revise = f"""
## Task Description
Given the extraction results of a research paper, please determine if the main research problem is a scientific problem from traditional disciplines in Science (not including disciplines like Computer Science and Information Science), and if the main method involves the use of Artificial Intelligence. 

## Research Paper to Be Extracted 
Title: {title}
Abstract: {abstract}

## Extraction Results

{
    results
}

## Note
* Don't consider research problem disciplines that involves Computer Science, Information/Data Science, Mathematics, Optimization, Statistics as traditional scientific problems.

## Response Format: Only a dictionary containing the following
{{
    "Scientific problem": True/False,
    "AI method": True/False,
}}
"""
    return prompt_revise

def get_prompt_phase2_blank():
    prompt_revise = f"""## Task Description
Please determine if the main research problem is a scientific problem from traditional disciplines in Science (not including disciplines like Computer Science and Information Science), and if the main method involves the use of Artificial Intelligence. 

## Response Format
{{
    "Scientific problem": True/False,
    "AI method": True/False,
}}
"""
    return prompt_revise


def get_prompt_AI_prediction_linked_direct_test_oneshot(record,linked_words):
    #print(record)
    del record['Method (keyword/keyphrase)']
    del record['Method (definition)']
    del record['Usage']
    def dict_to_line(record):
        string = ""
        for i in record.keys():
            string += i
            string += ": "
            string += record[i]
            string += "\n"
        return string
    def linked_words_to_line(linked_words):
        string = ""
        for i in range(len(linked_words)):
            string += f"Keywords {i}"
            string += ": "
            string += ','.join(linked_words[i])
            string += "\n"
        return string
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem, your task is to recommend potential Artificial Intelligence methods that can be used to address this scientific problem.
A few keywords of potential AI methods that have been statistically shown to address similar scientific problems will be given.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 
* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 
* Usage: How the method is specifically applied to address the problem. 

## Scientific Problem
Please recommend an AI method to address the below scientific problem for writing an academic paper:

{dict_to_line(record)}

## Potential AI methods

{linked_words_to_line(linked_words)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* Please respond with specific AI methods instead of high-level AI methods.
* Exactly output one recommendation.

## Response Format
Please output the recommendation of AI methods as a list, which has exactly *one* element. The output should be in the format of a list as below:

[
    {{
        "AI Method (keyword/keyphrase)": "...",
        "AI Usage": "..."
    }},
    ...
]"""
    return prompt_revise

def get_prompt_summarization_sci(top_words,records):
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if("Problem") in i:
                string += i
                string += ": "
                string += record[i]
                string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(records):
        examples += get_example(ind+1,example_record)
        examples += "\n"
    prompt_revise = f"""## Background and Task Description

You are an assistant in scientific research. Given a cluster of scientific problems/challenges, please help summarizing the cluster into a keyword or keyphrase less than 3 words. The top appearing words in the cluster together with some examples of scientific problem from that cluster will be provided. Please summarize based on the provided information.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 


## Top words from texual information

Below are some top words in we extracted from texual information of this cluster using TF-IDF

{','.join(top_words)}

## Examples of scientific papers

Below are some specific examples of scientific challenges in this cluster. For each sample the problem keyword, definition and discipline will be provided.

{examples}

## Summarization Requirement
Based on the above information, please find a keyword or a keyphrase less than *three* words to summarize this cluster of scientific problems, satisfying the following requirements:

* The summarization should be able to cover all the given top words and examples (i.e., with a high "recall")
* The summarization should be specific enough so that it only covers contents in this cluster (i.e., with a high "precision")
* If the summarization is not possible due to lack of information, output "N/A" when applicable.

## Response Format
Please output the summarization as a list, which has exactly *one* elements. The output should be in the format of a python list.

["Keyword"]

"""
    return prompt_revise

def get_prompt_summarization_ai(top_words,records):
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if("Method") in i:
                string += i
                string += ": "
                string += record[i]
                string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(records):
        examples += get_example(ind+1,example_record)
        examples += "\n"
    prompt_revise = f"""## Background and Task Description

You are an assistant in Artificial Intelligence research, where where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms. Given a cluster of Artificial Intelligence methods, please help summarizing the cluster into a keyword or keyphrase less than 3 words. The top appearing words in the cluster together with some examples of Artificial Intelligence metho from that cluster will be provided. Please summarize based on the provided information.

* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 


## Top words from texual information

Below are some top words in we extracted from texual information of this cluster using TF-IDF

{','.join(top_words)}

## Examples of Artificial Intelligence methods

Below are some specific examples of Artificial Intelligence methods in this cluster. For each sample the problem keyword, definition and discipline will be provided.

{examples}

## Summarization Requirement
Based on the above information, please find a keyword or a keyphrase less than *three* words to summarize this cluster of AI methods, satisfying the following requirements:

* The summarization should be able to cover all the given top words and examples (i.e., with a high "recall")
* The summarization should be specific enough so that it only covers contents in this cluster (i.e., with a high "precision")
* If the summarization is not possible due to lack of information, output "N/A" when applicable.

## Response Format
Please output the summarization as a list, which has exactly *one* elements. The output should be in the format of a python list.

["Keyword"]
"""
    return prompt_revise


def get_prompt_summarization_ai_near_clusters(top_words,records,near_top_words):
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if("Method") in i:
                string += i
                string += ": "
                string += record[i]
                string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(records):
        examples += get_example(ind+1,example_record)
        examples += "\n"
    def parse_near_top_words(near_top_words):
        words = ""
        for i in range(len(near_top_words)):
            words += f"Nearby cluster {i} with top words: {near_top_words[i][:20]}\n"
        return words
    # def title_abstract(titles,abstracts):
    #     examples = ""
    #     for i in range(len(titles)):
    #         examples += f"Paper {i}:\n Title: {titles[i]} \n Abstract: {abstracts[i]} \n\n"
    #     return examples 
    prompt_revise = f"""## Background and Task Description

You are an assistant in Artificial Intelligence research, where where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms. Given a cluster of Artificial Intelligence methods, please help summarizing the cluster into a keyword or keyphrase less than 3 words. The top appearing words in the cluster together with some examples of Artificial Intelligence methods from that cluster will be provided. Moreover, top words from several nearby clusters are provided and your summary should accurately capture the essence of the cluster while differentiating it from neighboring clusters.. Please summarize based on the provided information.

* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 


## Top words from texual information

Below are some top words in we extracted from texual information of this cluster using TF-IDF

{','.join(top_words)}

## Examples of Artificial Intelligence methods

Below are some specific examples of Artificial Intelligence methods in this cluster. For each sample the method keyword and definition will be provided.

{examples}

## Top words from nearby clusters' texual information

{parse_near_top_words(near_top_words)}

## Summarization Requirement
Based on the above information, please find a keyword or a keyphrase less than *three* words to summarize this cluster of AI methods, satisfying the following requirements:

* The summarization should be able to cover all the given top words and examples (i.e., with a high "recall")
* The summarization should be specific enough so that it only covers contents in this cluster (i.e., with a high "precision")
* If the summarization is not possible due to lack of information, output "N/A" when applicable.
* The summarization should be specific enough to differentiate it from nearby clusters
* Also provide summarization for nearby clusters

## Response Format
Please output the summarization as a list, which has exactly *(#nearby clusters)* outputs. The output should be in the format of a python list.

["Keyword", "Keyword neighbor 1",...]
"""
    return prompt_revise

def get_prompt_summarization_sci_near_clusters(top_words,records,near_top_words):
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if("Problem") in i or "Discipline" in i:
                string += i
                string += ": "
                string += record[i]
                string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(records):
        examples += get_example(ind+1,example_record)
        examples += "\n"
    def parse_near_top_words(near_top_words):
        words = ""
        for i in range(len(near_top_words)):
            words += f"Nearby cluster {i} with top words: {near_top_words[i][:20]}\n"
        return words
    # def title_abstract(titles,abstracts):
    #     examples = ""
    #     for i in range(len(titles)):
    #         examples += f"Paper {i}:\n Title: {titles[i]} \n Abstract: {abstracts[i]} \n\n"
    #     return examples 
    prompt_revise = f"""## Background and Task Description

You are an assistant in scientific research. Given a cluster of scientific problems/challenges, please help summarizing the cluster into a keyword or keyphrase less than 3 words. Moreover, top words from several nearby clusters are provided and your summary should accurately capture the essence of the cluster while differentiating it from neighboring clusters. Please summarize based on the provided information.

* Problem (keyword/keyphrase): A keyword or a keyphrase that summarizes the main problem to be addressed in this paper. 
* Problem (definition): The detailed definition of the problem. 
* Problem discipline: The discipline in which the main problem best fits. 


## Top words from texual information

Below are some top words in we extracted from texual information of this cluster using TF-IDF

{','.join(top_words)}

## Examples of Scientific Problems

Below are some specific examples of scientific problems in this cluster. For each sample the problem keyword, definition and discipline will be provided.

{examples}

## Top words from nearby clusters' texual information

{parse_near_top_words(near_top_words)}

## Summarization Requirement
Based on the above information, please find a keyword or a keyphrase less than *three* words to summarize this cluster of scientific problems, satisfying the following requirements:

* The summarization should be able to cover all the given top words and examples (i.e., with a high "recall")
* The summarization should be specific enough so that it only covers contents in this cluster (i.e., with a high "precision")
* If the summarization is not possible due to lack of information, output "N/A" when applicable.
* The summarization should be specific enough to differentiate it from nearby clusters
* Also provide summarization for nearby clusters

## Response Format
Please output the summarization as a list, which has exactly *(#nearby clusters+1)* outputs. The output should be in the format of a python list.

["Keyword", "Keyword neighbor 1",...]
"""
    return prompt_revise


def get_prompt_summarization_ai_near_clusters_paper(top_words,records,near_top_words,titles,abstracts):
    def dict_to_line_example(record):
        string = ""
        for i in record.keys():
            if("Method") in i:
                string += i
                string += ": "
                string += record[i]
                string += "\n"
        return string
    def get_example(ind,record):
        return f"""
        Example {ind}: \n
        {dict_to_line_example(record)}
        """
    examples = ""
    for ind, example_record in enumerate(records):
        examples += get_example(ind+1,example_record)
        examples += "\n"
    def parse_near_top_words(near_top_words):
        words = ""
        for i in range(len(near_top_words)):
            words += f"Nearby cluster {i} with top words: {near_top_words[i]}\n"
        return words
    def title_abstract(titles,abstracts):
        examples = ""
        for i in range(len(titles)):
            examples += f"Paper {i}:\n Title: {titles[i]} \n Abstract: {abstracts[i]} \n\n"
        return examples
    prompt_revise = f"""## Background and Task Description

You are an assistant in Artificial Intelligence research, where where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms. Given a cluster of Artificial Intelligence methods, please help summarizing the cluster into a keyword or keyphrase less than 3 words. The top appearing words in the cluster together with some examples of Artificial Intelligence metho from that cluster will be provided. Moreover, top words from several nearby clusters are provided and your summary should accurately capture the essence of the cluster while differentiating it from neighboring clusters.. Please summarize based on the provided information.

* Method (keyword/keyphrase): A keyword or a keyphrase that summarizes the main method used in this paper to address the above problem. 
* Method (definition): The detailed definition of the method. 


## Top words from texual information

Below are some top words in we extracted from texual information of this cluster using TF-IDF

{','.join(top_words)}

## Examples of papers belonging to that cluster

Below are some specific examples of papers in this cluster.

{title_abstract(titles,abstracts)}

## Top words from nearby clusters' texual information

{parse_near_top_words(near_top_words)}

## Summarization Requirement
Based on the above information, please find a keyword or a keyphrase less than *three* words to summarize this cluster of AI methods, satisfying the following requirements:

* The summarization should be able to cover all the given top words and examples (i.e., with a high "recall")
* The summarization should be specific enough so that it only covers contents in this cluster (i.e., with a high "precision")
* If the summarization is not possible due to lack of information, output "N/A" when applicable.
* The summarization should be specific enough to differentiate it from nearby clusters

## Response Format
Please output the summarization as a list, which has exactly *(#nearby clusters+1)* outputs. The output should be in the format of a python list.

["Keyword", "Keyword neighbor 1",...]
"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_at10(sci_cluster,AI_clusters):
    #print(record)
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains

{AI_clusters}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains
* Exactly recommend *ten* AI methods

## Response Format 
please output the recommendation it has used as a list, which contains exactly *ten* elements. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_at5(sci_cluster,AI_clusters):
    #print(record)
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains

{AI_clusters}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains
* Exactly recommend *five* AI methods

## Response Format 
please output the recommendation it has used as a list, which contains exactly *five* elements. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_at3(sci_cluster,AI_clusters):
    #print(record)
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains

{AI_clusters}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains
* Exactly recommend *three* AI methods

## Response Format 
please output the recommendation it has used as a list, which contains exactly *three* elements. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_at1(sci_cluster,AI_clusters):
    #print(record)
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains

{AI_clusters}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains
* Exactly recommend one AI method

## Response Format 
please output the recommendation it has used as a list, which contains exactly *one* element. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise


def get_prompt_AI_prediction_cluster_examples_at1(sci_cluster,AI_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains
{AI_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains.
* The AI methods recommended may or may not be within the given observed links.
* Exactly recommend *one* AI method

## Response Format 
please output the recommended AI methods as a list, which contains exactly *one* element. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_examples_at3(sci_cluster,AI_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains
{AI_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains.
* The AI methods recommended may or may not be within the given observed links.
* Exactly recommend *three* AI methods

## Response Format 
please output the recommended AI methods as a list, which contains exactly *three* elements. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_examples_at5(sci_cluster,AI_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains
{AI_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains.
* The AI methods recommended may or may not be within the given observed links.
* Exactly recommend *five* AI methods

## Response Format 
please output the recommended AI methods as a list, which contains exactly *five* elements. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_AI_prediction_cluster_examples_at10(sci_cluster,AI_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given a scientific problem domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential Artificial Intelligence methods that can be used to address the scientific problem.

## Scientific problem domain
{sci_cluster}

## Possible Artificial Intelligence domains
{AI_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential AI method can be used, mark it as "N/A" (not applicable). 
* The AI methods recommended should be within the possible Artificial Intelligence domains.
* The AI methods recommended may or may not be within the given observed links.
* Exactly recommend *ten* AI methods

## Response Format 
please output the recommended AI methods as a list, which contains exactly *ten* elements. The output should be in the list format as below:

[
    "Artificial Intelligence Method 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_examples_at1(AI_cluster,Sci_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* The scientifics problems recommended may or may not be within the given observed links.
* Exactly recommend *one* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *one* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_examples_at10(AI_cluster,Sci_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* The scientifics problems recommended may or may not be within the given observed links.
* Exactly recommend *ten* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *ten* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_examples_at3(AI_cluster,Sci_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* The scientifics problems recommended may or may not be within the given observed links.
* Exactly recommend *three* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *three* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_examples_at5(AI_cluster,Sci_clusters,example_links):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}

## Format of past usage of AI methods to solve Scientific problems:

(u,v,k): Scientific problem u has been solved with AI method v for k times in previous scientific literature.

## Past usage of AI methods to solve Scientific problems:
{parse_examples(example_links)}

## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* The scientifics problems recommended may or may not be within the given observed links.
* Exactly recommend *five* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *five* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_at1(AI_cluster,Sci_clusters):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}


## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* Exactly recommend *one* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *one* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_at3(AI_cluster,Sci_clusters):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}


## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* Exactly recommend *three* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *three* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_at5(AI_cluster,Sci_clusters):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}


## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* Exactly recommend *five* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *five* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_Sci_prediction_cluster_at10(AI_cluster,Sci_clusters):
    #print(record)
    def parse_examples(example_links):
        examples = ""
        for link in example_links:
            examples += f"({link[0]},{link[1]},{link[2]}); "
        return examples
    prompt_revise = f"""## Background and Task Description

You are an expert in both science and artificial intelligence (AI), where AI generally refers to intelligence exhibited by machines (particularly computer systems), including models, algorithms, etc. Given an AI method domain and past usage of Artificial Intelligence methods in solving scientific problems, your task is to recommend potential scientific problems that can be addressed by the AI method.

## AI method domain
{AI_cluster}

## Possible scientific problem domains
{Sci_clusters}


## Notes 
* If no potential scientific problem can be solved, mark it as "N/A" (not applicable). 
* The scientifics problems recommended should be within the possible scientific problem domains.
* Exactly recommend *ten* scientific problems.

## Response Format 
please output the recommended scientific problems as a list, which contains exactly *ten* elements. The output should be in the list format as below:

[
    "scientific problem 1",
    ...
]"""
    return prompt_revise

def get_prompt_summarization_sci_phase2(cluster_names):
        prompt_revise = f"""## Background and Task Description
    You are an expert in science. Given a list of summarizations of scientific problems addressed in a group of scientific papers that share similar scientific problems, you are asked to provide a summarization that best describes this group of scientific papers.


    ## Summarizations
    {cluster_names}

    ## Summarization Requirement
    Based on the above summarizations, please find a keyword or a keyphrase less than *three* words to summarize this cluster of scientific problems, satisfying the following requirements:

    ## Response Format
    Please output the summarization as a list, which has exactly *one* elements. The output should be in the format of a python list.

    ["Keyword"]

    """