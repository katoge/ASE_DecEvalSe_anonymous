# ASE_DecEvalSe_anonymous
DecEvalSe_Research_Result_Verification

We tend to publish the source code of the framework after acceptance, since otherwise it can give other researchers an advantage to publish results associated with this framework.
Nevertheless, we provide the "outputs" of the tool, and provide our anonymous reviewers some Python scripts to easily verify all results. Any output can be verified manually.

This repository serves the purpose to validate any research results used in Tables and Figures of our paper. We give in the discussion section of our paper a few words about verifiability.

The temporary scripts were generated by ChatGPT.

Usage:
0. Choose a Figure or Table that you want to validate.
1. Navigate to the respective folder in this repository. You will find multiple files. function_logs.jsonl, edit_distances.txt, outputs.txt are produced by the framework. Everything else are helper scripts generated by ChatGPT to let you verify the results.
2. Execute "python3 Statistics"
   This will give you information about compile ration, pass ratio, Levensthein Similarity, 
4. Execute "python3 tmp.py"
   This uses official libraries to calculate any other metrics like CodeBleu, CodeBertScore, etc.
   Be careful, for 1500 functions it can take a lot of time, when not using appropiate ressources.

These scripts merely collect data from function_logs.jsonl and edit_distances.txt, which are produced from the framework, and compute some metrics or statistics.

function_logs.jsonl contains functions and predictions/decompilations by LLM4Decompile and ANGR.
edit_distances.txt contain relevant code similarity metrics.

Used Decompilers:
https://github.com/albertan017/LLM4Decompile/tree/main
https://angr.io/

CodeAlign:
https://github.com/squaresLab/codealign/tree/main

